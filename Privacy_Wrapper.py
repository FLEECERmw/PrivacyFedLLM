from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from opacus.data_loader import DPDataLoader
import torch.distributed as dist
from config import ExpConfig


import logging
logger = logging.getLogger(__name__)

def get_sigma(exp:ExpConfig, sample_num):
    ddp_proc = dist.get_world_size()
    sample_rate = exp.batch_size * ddp_proc / sample_num # total_bsz / len(local_dataset)
    steps = exp.num_communication_rounds * exp.num_local_steps * (exp.num_clients_per_round / exp.num_clients) # num_comm_rounds * num_local_Updates * client_sample_ratio
    dp_config = exp.privacy

    return get_noise_multiplier(
            target_epsilon=dp_config.target_epsilon,
            target_delta=dp_config.target_delta,
            sample_rate=sample_rate, 
            steps=int(steps), 
            accountant="rdp",
        )


def distributed_privacy_wrapper(exp: ExpConfig, sample_num, local_rank, model, optimizer, dataloader):
    privacy_engine = PrivacyEngine()
    noise_multiplier = get_sigma(exp, sample_num)
    logger.info(f"Noise multiplier of DP: {noise_multiplier}")
    dp_config = exp.privacy

    private_dl = DPDataLoader.from_data_loader(dataloader, distributed=True)
    private_model = model

    dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
        module=private_model,
        optimizer=optimizer,
        data_loader=private_dl,
        noise_multiplier=noise_multiplier,
        max_grad_norm=dp_config.target_grad_clip_persample,
        poisson_sampling=True
    )
    return dp_model, dp_optimizer, dp_dataloader


def privacy_wrapper(exp: ExpConfig, sample_num, local_rank, model, optimizer, dataloader):
    privacy_engine = PrivacyEngine()
    noise_multiplier = get_sigma(exp, sample_num)
    dp_config = exp.privacy
    logger.info(f"Noise multiplier of DP: {noise_multiplier}")
    dp_model, dp_optimizer, dp_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=dp_config.target_grad_clip_persample
    )
    return dp_model, dp_optimizer, dp_dataloader
