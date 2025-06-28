import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import logging 
from .basepolicy import FedPolicy

module_logger = logging.getLogger(__name__)

def get_Omega(
    model_state_dict: Dict[str, torch.Tensor],
    s_sketch_dim: int,
    target_device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, torch.Tensor]:
    """
    Creates auxiliary random matrices Omega for LoRA layers based on model_state_dict.
    Omega is used for sketching. LoRA layers are identified by '.lora_A.weight'.
    """
    log = logger if logger is not None else module_logger
    global_auxiliary_omegas = {}

    for name, param in model_state_dict.items():
        if '.lora_A.weight' in name:
            _lora_rank, k_features = param.shape
            current_device = target_device if target_device is not None else param.device
            Omega = torch.randn(k_features, s_sketch_dim, device=current_device)
            base_layer_name = name.replace('.lora_A.weight', '')
            global_auxiliary_omegas[base_layer_name] = Omega

    if not global_auxiliary_omegas:
        log.warning("No LoRA A weights found in model_state_dict. Omega dictionary will be empty.")
    return global_auxiliary_omegas


class FedASKPolicy(FedPolicy):
    def __init__(
        self,
        initial_global_model_state_dict: Dict[str, torch.Tensor],
        s_sketch_dim: int,
        sample_num_dict: Dict[Any, Union[int, float]],
        lora_rank: int,
        target_device: Optional[torch.device] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs
    ):
        """
        Initializes the FedASKPolicy.

        Args:
            initial_global_model_state_dict: State dictionary of the initial global model.
            s_sketch_dim: The dimension 's' for the sketch matrices Omega.
            sample_num_dict: Dictionary mapping client ID to their number of samples.
            lora_rank: Target rank for reconstructed LoRA layers.
            target_device: Optional torch.device for tensor creation.
            logger: Optional logger instance.
        """
        self.log = logger if logger is not None else module_logger
        
        self.s_sketch_dim = s_sketch_dim
        # Pass self.log to get_Omega so it uses the same logger instance
        self.Omega_global_dict = get_Omega(initial_global_model_state_dict, s_sketch_dim, target_device, logger=self.log)
        self.sample_num_dict = sample_num_dict
        self.lora_rank = lora_rank
        
        if self.Omega_global_dict:
            self.device = next(iter(self.Omega_global_dict.values())).device
        elif initial_global_model_state_dict:
            self.device = next(iter(initial_global_model_state_dict.values())).device
        else:
            self.device = target_device if target_device is not None else torch.device("cpu")
        
        for name in self.Omega_global_dict: # Ensure device consistency
            self.Omega_global_dict[name] = self.Omega_global_dict[name].to(self.device)

    def local_transfer_stage_1(
        self,
        local_model_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Client-side computation for Stage 1: Y_i = W_B_i @ (W_A_i @ Omega_global)"""
        Y_i_sketch = {}
        for layer_name, Omega_val in self.Omega_global_dict.items():
            lora_B_weight = local_model_dict[f"{layer_name}.lora_B.weight"].to(self.device)
            lora_A_weight = local_model_dict[f"{layer_name}.lora_A.weight"].to(self.device)
            Y_i_sketch[layer_name] = lora_B_weight @ (lora_A_weight @ Omega_val)
        return Y_i_sketch

    def local_transfer_stage_2(
        self,
        local_model_dict: Dict[str, torch.Tensor],
        global_Q_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Client-side computation for Stage 2: P_i = W_A_i^T @ (W_B_i^T @ Q_global)"""
        P_i_sketch = {}
        for layer_name, Q_val in global_Q_dict.items():
            lora_A_weight = local_model_dict[f"{layer_name}.lora_A.weight"].to(self.device)
            lora_B_weight = local_model_dict[f"{layer_name}.lora_B.weight"].to(self.device)
            Q_val = Q_val.to(self.device)
            P_i_sketch[layer_name] = lora_A_weight.T @ (lora_B_weight.T @ Q_val)
        return P_i_sketch
    
    def global_aggregate(
        self,
        local_model_dicts: Dict[Any, Dict[str, torch.Tensor]],
        clients_this_round: List[Any]
    ) -> Dict[str, torch.Tensor]:
        """Server-side aggregation of client updates using the two-stage FedASK mechanism."""
        if not clients_this_round:
            self.log.warning("No clients provided for aggregation. Returning empty dictionary.")
            return {}

        client_sample_counts = torch.tensor(
            [self.sample_num_dict.get(client_idx, 0) for client_idx in clients_this_round],
            dtype=torch.float32, device=self.device
        )
        total_samples_this_round = client_sample_counts.sum()

        if total_samples_this_round == 0:
            self.log.warning("Total samples for participating clients is 0. Cannot perform weighted aggregation.")
            return {}
        client_weights = client_sample_counts / total_samples_this_round

        client_Y_sketches_this_round: Dict[Any, Dict[str, torch.Tensor]] = {}
        for client_idx in clients_this_round:
            client_Y_sketches_this_round[client_idx] = self.local_transfer_stage_1(local_model_dicts[client_idx])
        
        global_Q_dict: Dict[str, torch.Tensor] = {}
        for layer_name in self.Omega_global_dict.keys():
            example_Y_for_shape = None
            for client_idx in clients_this_round:
                if layer_name in client_Y_sketches_this_round.get(client_idx, {}):
                    example_Y_for_shape = client_Y_sketches_this_round[client_idx][layer_name]
                    break
            if example_Y_for_shape is None:
                self.log.warning(f"Layer '{layer_name}' not in any client's Y_sketch. Skipping Stage 1 for this layer.")
                continue
            
            Y_aggregated_layer = torch.zeros_like(example_Y_for_shape)
            for i, client_idx in enumerate(clients_this_round):
                if layer_name in client_Y_sketches_this_round.get(client_idx, {}):
                    Y_aggregated_layer += client_Y_sketches_this_round[client_idx][layer_name] * client_weights[i]
            Q_layer, _R_layer = torch.linalg.qr(Y_aggregated_layer)
            global_Q_dict[layer_name] = Q_layer

        client_P_sketches_this_round: Dict[Any, Dict[str, torch.Tensor]] = {}
        for client_idx in clients_this_round:
            client_P_sketches_this_round[client_idx] = self.local_transfer_stage_2(
                local_model_dicts[client_idx], global_Q_dict
            )

        reconstructed_global_lora_weights: Dict[str, torch.Tensor] = {}
        for layer_name in self.Omega_global_dict.keys():
            if layer_name not in global_Q_dict:
                self.log.warning(f"Global Q for layer '{layer_name}' not computed. Skipping Stage 2 reconstruction.")
                continue

            example_P_for_shape = None
            for client_idx in clients_this_round:
                if layer_name in client_P_sketches_this_round.get(client_idx, {}):
                    example_P_for_shape = client_P_sketches_this_round[client_idx][layer_name]
                    break
            if example_P_for_shape is None:
                self.log.warning(f"Layer '{layer_name}' not in any client's P_sketch. Skipping Stage 2 for this layer.")
                continue
            
            P_aggregated_layer = torch.zeros_like(example_P_for_shape)
            for i, client_idx in enumerate(clients_this_round):
                 if layer_name in client_P_sketches_this_round.get(client_idx, {}):
                    P_aggregated_layer += client_P_sketches_this_round[client_idx][layer_name] * client_weights[i]
            U, S_vec, Vh = torch.linalg.svd(P_aggregated_layer.T, full_matrices=False)
            
            rank_to_use = min(self.lora_rank, S_vec.size(0))
            if rank_to_use < self.lora_rank:
                self.log.warning(f"Requested lora_rank {self.lora_rank} > available singular values {S_vec.size(0)} for layer '{layer_name}'. Using rank {rank_to_use}.")

            sqrt_S_r = torch.sqrt(S_vec[:rank_to_use])
            lora_B_factor = U[:, :rank_to_use] * sqrt_S_r
            
            reconstructed_global_lora_weights[f"{layer_name}.lora_B.weight"] = global_Q_dict[layer_name] @ lora_B_factor
            reconstructed_global_lora_weights[f"{layer_name}.lora_A.weight"] = sqrt_S_r.unsqueeze(1) * Vh[:rank_to_use, :]

        return reconstructed_global_lora_weights