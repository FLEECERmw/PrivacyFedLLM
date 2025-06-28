import logging
import copy
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)
from trl import DataCollatorForCompletionOnlyLM
from torch.utils.data import DataLoader, DistributedSampler
from peft import (
    get_peft_model,
    LoraConfig as PeftLoraConfig,
    prepare_model_for_kbit_training,
)

from data import get_dataset, sample_dataset, split_dataset, get_formatting_prompts_func
from policy.router import get_policy
from config import ExpConfig, DataConfig

logger = logging.getLogger(__name__)


class AssetBuilder:
    def __init__(self, exp: ExpConfig, is_distributed: bool, rank: int, world_size: int):
        self.exp_config = exp
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = None

    def build_model_and_tokenizer(self, device: str):
        exp = self.exp_config
        logger.info(f"Building model: {exp.name_model} (Quant: {exp.mode_quant}, Precision: {exp.model_precision})")
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(exp.model_precision, torch.float32)
        
        load_kwargs = {"trust_remote_code": True, "torch_dtype": torch_dtype}
        if exp.mode_quant == "8bit":
            load_kwargs.update({
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True), 
                "device_map": device,
            })
        elif exp.mode_quant == "4bit":
            load_kwargs.update({
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, 
                    bnb_4bit_use_double_quant=True, 
                    bnb_4bit_quant_type="nf4", 
                    bnb_4bit_compute_dtype=torch_dtype
                ), 
                "device_map": device,
            })

        model = AutoModelForCausalLM.from_pretrained(exp.name_model, **load_kwargs)
        if exp.mode_quant in ["8bit", "4bit"]:
            model = prepare_model_for_kbit_training(model)

        tokenizer = AutoTokenizer.from_pretrained(exp.name_model, use_fast=False, padding_side="right")
        if tokenizer.pad_token is None: # Dynamically set pad_token if not present
            name_lower = exp.name_model.lower()
            pad_set_success = False
            if "llama3" in name_lower or "llama-3" in name_lower:
                if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token; pad_set_success=True
            elif "llama2" in name_lower or "llama-2" in name_lower:
                if tokenizer.unk_token: tokenizer.pad_token = tokenizer.unk_token; pad_set_success=True
            elif tokenizer.eos_token: 
                tokenizer.pad_token = tokenizer.eos_token; pad_set_success=True
            if not pad_set_success or tokenizer.pad_token is None:
                raise ValueError(f"Pad token could not be set for tokenizer: {exp.name_model}")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            logger.info(f"Set pad_token to '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
        self.tokenizer = tokenizer

        if hasattr(exp, 'lora_config') and exp.lora_config:
            lora_config = PeftLoraConfig(
                r=exp.lora_config.r,
                lora_alpha=exp.lora_config.lora_alpha,
                target_modules=exp.lora_config.target_modules,
                lora_dropout=exp.lora_config.lora_dropout,
                bias=exp.lora_config.bias,
                task_type=exp.lora_config.task_type,
            )
            model = get_peft_model(model, lora_config)
            if hasattr(exp.policy, 'freezeA') and exp.policy.freezeA:
                for name, param in model.named_parameters():
                    if "lora_A" in name: param.requires_grad = False
                logger.info("Froze LoRA 'A' matrices.")
            model.print_trainable_parameters()

        if exp.enable_gradient_checkpoint and not exp.privacy.use_privacy:
            if model.supports_gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Gradient checkpointing enabled.")
            else:
                logger.warning("Model does not support gradient checkpointing, skipping.")
        elif exp.enable_gradient_checkpoint and exp.privacy.use_privacy:
            logger.warning("Gradient checkpointing is incompatible with Opacus, so it has been disabled.")
                    
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
            
        return model, tokenizer

    def build_optimizer(self, model_to_optimize: torch.nn.Module):
        exp = self.exp_config
        logger.info("Building optimizer.")
        opt_cfg = exp.optimizer_config
        trainable_params = filter(lambda p: p.requires_grad, model_to_optimize.parameters())
        return torch.optim.AdamW(
            trainable_params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
        )

    def build_dataloaders(self):
        exp = self.exp_config
        dataconfig = exp.data
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized. Call build_model_and_tokenizer first.")
            
        logger.info(f"Building dataloaders for {exp.num_clients} clients from '{dataconfig.name_dataset}'.")
        dataset = get_dataset(dataconfig.name_dataset, dataconfig.name_localdatadir)
        dataset = sample_dataset(dataset, dataconfig.num_sample_all)
        local_client_datasets = split_dataset(
            dataset, exp.num_clients, dataconfig.mode_datadist, dataconfig.para_niid_deg
        )
        sample_num_dict = {i: len(ds) for i, ds in enumerate(local_client_datasets) if ds}
        
        formatting_func, response_template_str = get_formatting_prompts_func(
            dataconfig.template, self.tokenizer.eos_token
        )
        
        encoded_resp_ids = self.tokenizer.encode(response_template_str, add_special_tokens=False)
        is_leading_special = len(encoded_resp_ids) > 1 and \
                             (self.tokenizer.decode(encoded_resp_ids[0]).strip() == "" or \
                              encoded_resp_ids[0] == self.tokenizer.bos_token_id)
        encoded_resp_ids = encoded_resp_ids[1:] if is_leading_special else encoded_resp_ids # Slice if needed
        
        trl_collator = DataCollatorForCompletionOnlyLM(
            response_template=encoded_resp_ids, tokenizer=self.tokenizer
        )
        data_collator = trl_collator

        client_dataloaders = {}
        for idx, client_ds in enumerate(local_client_datasets):
            if not client_ds: continue
            
            tokenized_ds = client_ds.map(
                lambda batch: {"text": formatting_func(batch)},
                batched=True,
                remove_columns=client_ds.column_names
            ).map(
                lambda batch: self.tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=dataconfig.max_seq_length
                ),
                batched=True,
                remove_columns=["text"]
            )
            
            sampler = DistributedSampler(
                tokenized_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            ) if self.is_distributed else None
            
            client_dataloaders[idx] = DataLoader(
                tokenized_ds,
                batch_size=exp.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=dataconfig.num_workers,
                pin_memory=True,
                collate_fn=data_collator,
                drop_last=True,
            )
        
        logger.info(f"DataLoader construction finished. {len(client_dataloaders)} active DataLoaders created.")
        return client_dataloaders, sample_num_dict

    def build_policy(self, policy_name: str, policy_specific_init_config: dict):
        logger.info(f"Building policy: {policy_name}")
        return get_policy(policy_name, policy_specific_init_config)