import logging
import os
import json
import random
import copy
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from opacus.data_loader import DPDataLoader
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

from builder import AssetBuilder
from localtrainer import LocalTrainer
from config import ExpConfig
from Privacy_Wrapper import privacy_wrapper, distributed_privacy_wrapper 

logger = logging.getLogger(__name__)

def setup_ddp(backend="nccl"):
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=backend, init_method="env://")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def cleanup():
    dist.destroy_process_group()

def apply_global_seed(seed_value: int):
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class FederatedSftTrainer:
    def __init__(self, exp: ExpConfig):
        self.exp_config = exp
        self.rank, self.world_size, self.local_rank = setup_ddp()
        self.is_distributed = self.world_size > 1
        self.is_private = exp.privacy.use_privacy
        self.device = torch.device(f"cuda:{self.local_rank}")
        logger.info(f"[Rank {self.rank}] Using device: {self.device}, total world size{self.world_size}")

        builder = AssetBuilder(exp, self.is_distributed, self.rank, self.world_size)
        self.model, self.tokenizer = builder.build_model_and_tokenizer(str(self.device))
        self.model.to(self.device)
        self.clients_dataloader, self.sample_num_dict = builder.build_dataloaders()

        if self.is_distributed:
            if self.is_private:
                from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
                self.model = DPDDP(self.model)
                logger.info("Wrapped model with Opacus DPDDP.")
            else:
                find_unused = getattr(exp, "ddp_find_unused_parameters", False)
                self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=find_unused)
                logger.info("Wrapped model with standard DDP.")

        self.optimizer = builder.build_optimizer(self.model)

        if self.is_private:
            logger.info("Converting all client dataloaders to DPDataLoaders at initialization.")
            dp_dataloaders = {}
            for i, loader in self.clients_dataloader.items():
                if loader:
                    dp_dataloaders[i] = DPDataLoader.from_data_loader(
                        loader, distributed=self.is_distributed
                    )
            self.clients_dataloader = dp_dataloaders

        if self.is_private:
            sample_num = min(self.sample_num_dict.values())
            wrapper_fn = distributed_privacy_wrapper if self.is_distributed else privacy_wrapper
            self.model, self.optimizer, _ = wrapper_fn(
                exp, sample_num, self.local_rank, self.model, self.optimizer, self.clients_dataloader[0]
            )
            logger.info("Applied custom privacy wrapper.")
        
        policy_init_cfg = copy.deepcopy(exp.policy.init_config)
        policy_init_cfg['sample_num_dict'] = self.sample_num_dict
        policy_init_cfg['target_device'] = self.device
        if exp.policy.name.lower() == "fedask":           
            policy_init_cfg['initial_global_model_state_dict'] = get_peft_model_state_dict(self.unwrapped_model)
        elif exp.policy.name.lower() in ["fedprox", "scaffold"]:
            policy_init_cfg['global_state'] = get_peft_model_state_dict(self.unwrapped_model)
        self.policy = builder.build_policy(exp.policy.name, policy_init_cfg)
        

        self.local_worker = LocalTrainer(
            exp_config=exp,
            device=self.device,
            is_distributed=self.is_distributed,
            rank=self.rank,
            policy=self.policy
        )

        if self.rank == 0 and self.exp_config.checkpoint_dir:
            os.makedirs(self.exp_config.checkpoint_dir, exist_ok=True)
        self.training_history = []

    @property
    def unwrapped_model(self):
        if hasattr(self.model, 'module'): return self.model.module
        if hasattr(self.model, '_module'): return self.model._module
        return self.model

    def train(self):
        exp_cfg = self.exp_config
        logger.info(f"Starting FL: Rounds={exp_cfg.num_communication_rounds}, Clients/Round={exp_cfg.num_clients_per_round}/{exp_cfg.num_clients}")

        for comm_round in range(exp_cfg.num_communication_rounds):
            logger.info(f"Comm Round {comm_round + 1}/{exp_cfg.num_communication_rounds}")
            
            chosen_clients_obj = [None]
            if self.rank == 0:
                eligible_clients = list(self.clients_dataloader.keys())
                num_to_select = min(exp_cfg.num_clients_per_round, len(eligible_clients))
                chosen_clients_obj[0] = random.sample(eligible_clients, num_to_select)
                logger.info(f"Rank 0 selected clients: {chosen_clients_obj[0]}")
            if self.is_distributed:
                dist.broadcast_object_list(chosen_clients_obj, src=0)
            
            chosen_client_indices = chosen_clients_obj[0]
            if not chosen_client_indices:
                logger.warning("No clients selected, skipping round.")
                continue

            local_outputs_for_agg = {}
            current_global_peft_state = get_peft_model_state_dict(self.unwrapped_model)
            for client_idx in chosen_client_indices:
                # init_peft_state = policy.init_(current_global_peft_state) # might need for fedsa
                set_peft_model_state_dict(self.unwrapped_model, copy.deepcopy(current_global_peft_state))
                
                client_dataloader = self.clients_dataloader[client_idx]
                if self.is_private:
                    num_samples = self.sample_num_dict.get(client_idx)
                    if num_samples is None:
                        num_samples = len(client_dataloader.dataset)

                history = self.local_worker.train_one_client(
                    self.model, client_dataloader, self.optimizer, comm_round, client_idx
                )
                
                self.training_history.extend(history)
                local_outputs_for_agg[client_idx] = copy.deepcopy(get_peft_model_state_dict(self.unwrapped_model))
    
            
            aggregated_params_obj = [None]
            if self.rank == 0 and local_outputs_for_agg:
                aggregated_params_obj[0] = self.policy.global_aggregate(
                    local_outputs_for_agg, chosen_client_indices
                )

            if self.is_distributed:
                dist.broadcast_object_list(aggregated_params_obj, src=0)
            
            aggregated_peft_params = aggregated_params_obj[0]
            if aggregated_peft_params:
                set_peft_model_state_dict(self.unwrapped_model, aggregated_peft_params)
                logger.info("Global PEFT model updated.")
            else:
                logger.warning("Aggregation resulted in no parameters. Global model not updated.")

            if self.rank == 0 and exp_cfg.checkpoint_dir and (comm_round + 1) % exp_cfg.save_every_n_rounds == 0:
                self._save_checkpoint(comm_round)

        logger.info("Federated training finished.")

    def _save_checkpoint(self, comm_round):
        exp_cfg = self.exp_config
        round_ckpt_dir = os.path.join(exp_cfg.checkpoint_dir, f"round_{comm_round+1}")
        os.makedirs(round_ckpt_dir, exist_ok=True)
        
        with open(os.path.join(round_ckpt_dir, "training_history.json"), 'w') as f:
            json.dump(self.training_history, f, indent=4)
        
        self.unwrapped_model.save_pretrained(os.path.join(round_ckpt_dir, "lora"))
        
        OmegaConf.save(config=exp_cfg, f=os.path.join(round_ckpt_dir, "experiment_config.yaml"))
        logger.info(f"Saved checkpoint for round {comm_round+1} to {round_ckpt_dir}")

def main():
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    cli_args = OmegaConf.from_cli()
    config_file = cli_args.pop('config_path', "config.yaml")
    yaml_conf = OmegaConf.load(config_file)
    schema = OmegaConf.structured(ExpConfig)
    conf = OmegaConf.merge(schema, yaml_conf, cli_args)
    exp_config_obj: ExpConfig = OmegaConf.to_object(conf)
    
    apply_global_seed(getattr(exp_config_obj, 'seed', 42))

    trainer = FederatedSftTrainer(exp=exp_config_obj)
    trainer.train()
    cleanup()
    
    if int(os.environ.get("RANK", 0)) == 0:
        print("\n--- Final Configuration ---")
        print(OmegaConf.to_yaml(conf))

if __name__ == "__main__":
    main()