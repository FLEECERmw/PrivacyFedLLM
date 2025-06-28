from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import OmegaConf

@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]) # Common for Llama
    lora_dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    task_type: str = "CAUSAL_LM"

@dataclass
class OptimizerConfig:
    # Parameters for torch.optim.AdamW
    lr: float = 2e-5
    weight_decay: float = 0.0
    betas: Tuple[float, float] = field(default_factory=lambda: (0.9, 0.999))
    eps: float = 1e-8

@dataclass
class SchedulerConfig:
    name: str = "linear"  # e.g., "linear", "cosine", "constant_with_warmup"
    num_warmup_steps: int = 0

@dataclass
class DataConfig:
    name_dataset: str = "databricks/databricks-dolly-15k"
    name_localdatadir: Optional[str] = None 
    num_sample_all: int = 10000 # Total samples to use from the dataset before splitting
    mode_datadist: str = "iid"  # "iid" or "dirichlet-noniid"
    para_niid_deg: Optional[float] = 0.5  
    template: str = "alpaca"  
    max_seq_length: int = 512
    num_workers: int = 0  

@dataclass
class PolicyConfig: # Renamed to avoid conflict with any PolicyConfig class definition
    name: str = "fedavg"  # Name of the policy, e.g., "fedask", "fedavg"
    freezeA: bool = False
    # `init_config` will hold parameters specific to the __init__ of the chosen policy class.
    # These are parameters beyond what's programmatically injected (like sample_num_dict, target_device).
    init_config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PrivacyConfig:
    use_privacy: bool = False
    target_epsilon: float = 6.0
    target_delta: float = 1e-5
    target_grad_clip_persample: float = 1.0


# --- Main Experiment Configuration ---
@dataclass
class ExpConfig:
    # Model and Tokenizer
    name_model: str = "NousResearch/Llama-2-7b-hf"
    mode_quant: Optional[str] = None  # "8bit", "4bit", or None
    model_precision: str = "bf16"  # "fp32", "fp16", "bf16"

    # Federated Learning Setup
    num_communication_rounds: int = 200
    num_clients: int = 20 # Total number of clients available
    num_clients_per_round: int = 5 # Number of clients selected each round
    num_local_steps: int = 10 # Local training steps per client per round

    # Training Hyperparameters
    batch_size: int = 4 # Batch size for client DataLoaders
    max_grad_norm: float = 1.0 # clip gradient norm
    gradient_accumulation_steps: int = 1 # gradient accumulation_steps
    ddp_find_unused_parameters: bool = False
    enable_gradient_checkpoint: bool = True

    # Saving Config
    checkpoint_dir: str|None = None
    save_every_n_rounds: int = 5
    log_every_n_steps: int = 2

    # Nested Configurations
    lora_config: LoraConfig = field(default_factory=LoraConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig) # Matches exp.data usage
    policy: PolicyConfig = field(default_factory=PolicyConfig) # Matches exp.policy usage
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)

    # General
    seed: int = 42

def load_config(yaml_path: str) -> ExpConfig:
    setup_omegaconf_resolvers()
    schema = OmegaConf.structured(ExpConfig)
    conf = OmegaConf.load(yaml_path)
    merged_conf = OmegaConf.merge(schema, conf)
    return OmegaConf.to_object(merged_connf) 