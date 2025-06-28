# local_train.py

import logging
import time
import torch
import torch.nn.utils
from tqdm.auto import tqdm
from transformers import get_scheduler
from config import ExpConfig

logger = logging.getLogger(__name__)

class LocalTrainer:
    def __init__(self, exp_config: ExpConfig, device: torch.device, is_distributed: bool, rank: int, policy):
        self.exp_config = exp_config
        self.device = device
        self.is_distributed = is_distributed
        self.rank = rank
        self.policy = policy
        self.grad_accum_steps = getattr(exp_config, 'gradient_accumulation_steps', 1)
        if not isinstance(self.grad_accum_steps, int) or self.grad_accum_steps < 1:
            self.grad_accum_steps = 1

    def _build_local_scheduler(self, optimizer):
        exp = self.exp_config
        sched_cfg = exp.scheduler_config
        num_optimizer_steps = (exp.num_local_steps + self.grad_accum_steps - 1) // self.grad_accum_steps
        return get_scheduler(
            name=sched_cfg.name,
            optimizer=optimizer,
            num_warmup_steps=sched_cfg.num_warmup_steps,
            num_training_steps=num_optimizer_steps
        )

    def train_one_client(self, model, dataloader, optimizer, comm_round, client_idx):
        model.train()
        local_scheduler = self._build_local_scheduler(optimizer)
        data_iter = iter(dataloader)
        
        pbar_desc = f"R:{comm_round+1}|C:{client_idx}"
        local_steps_pbar = tqdm(
            range(self.exp_config.num_local_steps),
            desc=pbar_desc,
            unit="step",
            leave=False,
            disable=(self.is_distributed and self.rank != 0)
        )
        
        training_history_for_client = []
        grad_norm_tensor = torch.tensor(0.0)

        for step in local_steps_pbar:
            start_time = time.time()
            try:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
            except Exception as e: 
                '''
                opacus might sample an empty batch due to possion sampling
                but the collate function will return a list of tensor which might raise dtype error when dealing dict item
                '''
                logger.warning(
                    f"Skipping local step {step+1} for client {client_idx} due to a critical data loading error: {e}"
                )
                continue 
            
            inputs = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            loss = self.policy.local_compute_loss(model, inputs)
            if self.grad_accum_steps > 1:
                loss = loss / self.grad_accum_steps
            
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0 or (step + 1) == self.exp_config.num_local_steps:
                if not self.exp_config.privacy.use_privacy:
                    max_norm = getattr(self.exp_config, 'max_grad_norm', float('inf'))
                else:
                    max_norm = float('inf') # dp has done the clipping
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()), max_norm
                )
                
                optimizer.step()
                optimizer.zero_grad()
                local_scheduler.step()

                if hasattr(self.policy, 'local_callback'):
                    self.policy.local_callback(
                        model=model,
                        client_id=client_idx,
                        learning_rate=local_scheduler.get_lr(),
                        step=step,
                        max_train_steps=self.exp_config.num_local_steps
                    )
            
            step_time = time.time() - start_time
            loss_val = loss.item() * self.grad_accum_steps
            grad_norm_val = grad_norm_tensor.item()
            
            if self.rank == 0:
                training_history_for_client.append({
                    'round': comm_round + 1, 'client': client_idx, 'step': step + 1,
                    'loss': loss_val, 'grad_norm': grad_norm_val, 'step_time_s': step_time
                })
                local_steps_pbar.set_postfix(loss=f"{loss_val:.4f}", grad_norm=f"{grad_norm_val:.2f}")

        return training_history_for_client