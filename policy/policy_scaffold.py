import torch
from collections import defaultdict # Not strictly needed here but often useful
from typing import Dict, List, Any, Optional, Union
import logging
from .basepolicy import FedPolicy
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import copy

module_logger = logging.getLogger(__name__)

def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Recursively unwraps a model from common wrappers like DDP and Opacus.
    """
    if hasattr(model, 'module'):
        return unwrap_model(model.module)
    if hasattr(model, '_module'):
        return unwrap_model(model._module)
    return model

class ScaffoldPolicy(FedPolicy):
    def __init__(
        self,
        sample_num_dict: Dict[Any, Union[int, float]],
        global_state: Dict[Any, torch.Tensor],
        logger: Optional[logging.Logger] = None,
        target_device: Optional[torch.device] = None,
        **kwargs # Added target_device
    ):
        self.log = logger if logger is not None else module_logger
        self.sample_num_dict = sample_num_dict
        self.global_state = global_state
        self.global_c = {
            name: torch.zeros_like(param)
            for name, param in self.global_state.items()
        }
        local_template = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.global_state.items()
        }
        self.local_c = {
            client_id: copy.deepcopy(local_template)
            for client_id in self.sample_num_dict.keys()
        }
        self.local_upd = {
            client_id: copy.deepcopy(local_template)
            for client_id in self.sample_num_dict.keys()
        }
        if target_device is not None:
            self.device = target_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.log.info(f"Initialized SCAFFOLD control variates for {len(self.sample_num_dict)} clients on CPU.")
    

    def local_callback(self, model, client_id, learning_rate, step, max_train_steps=10):
        unwrapped_model = unwrap_model(model)
        model_para = copy.deepcopy(get_peft_model_state_dict(unwrapped_model))
        for name in model_para.keys():
            try:
                model_para[name] -= learning_rate[0] * (self.global_c[name] - self.local_c[client_id][name].to(self.device))
            except:
                import torch.distributed as dist
                if dist.get_rank() == 0:
                    breakpoint()
                dist.barrier()

        new_local_c={}
        if step+1 == max_train_steps: # 更新self.local_c[idx]， self.local_upd[idx]
            with torch.no_grad():
                for name, param in model_para.items():
                    new_local_c[name] = (self.global_state[name] - param) / (learning_rate[0] * max_train_steps) - (self.global_c[name] - self.local_c[client_id][name].to(self.device))
                    self.local_upd[client_id][name] = (new_local_c[name] - self.local_c[client_id][name].to(self.device))
            self.local_c[client_id] = {name: tensor.to("cpu") for name, tensor in new_local_c.items()}
        set_peft_model_state_dict(unwrapped_model, model_para)
        
        return model


    def global_aggregate(
        self,
        local_model_dicts: Dict[Any, Dict[str, torch.Tensor]], # Key: client_id
        clients_this_round: List[Any] # List of client_ids participating
    ) -> Dict[str, torch.Tensor]:

        if not clients_this_round:
            self.log.warning("No clients provided for aggregation. Returning empty dictionary.")
            return {}

        # Ensure all participating clients are in sample_num_dict, default to 0 if not.
        client_sample_counts_list = []
        for client_idx in clients_this_round:
            count = self.sample_num_dict.get(client_idx, 0)
            if count == 0:
                self.log.warning(f"Client {client_idx} has 0 samples or not found in sample_num_dict. It will have 0 weight in aggregation.")
            client_sample_counts_list.append(count)

        client_sample_counts = torch.tensor(
            client_sample_counts_list,
            dtype=torch.float32, device=self.device
        )
        total_samples_this_round = client_sample_counts.sum()

        if total_samples_this_round == 0:
            self.log.warning("Total samples for participating clients is 0. Cannot perform weighted aggregation. Returning empty dictionary.")
            return {}
        
        # Weights for each client in clients_this_round, in the same order
        client_weights = client_sample_counts / total_samples_this_round
        
        global_dict: Dict[str, torch.Tensor] = {}
        
        # Get layer names from the first participating client's model
        # Assuming all clients have the same model structure (same layers)
        if not local_model_dicts or clients_this_round[0] not in local_model_dicts:
            self.log.error("local_model_dicts is empty or first client in clients_this_round not found in local_model_dicts.")
            return {}
            
        first_client_id = clients_this_round[0]
        layer_names = local_model_dicts[first_client_id].keys()

        for layer_name in layer_names:
            # Initialize aggregated_layer_tensor with zeros, using shape and device from the first client's layer
            # Ensure the example tensor is on the target device for aggregation
            try:
                example_tensor = local_model_dicts[first_client_id][layer_name].to(self.device)
            except KeyError:
                self.log.warning(f"Layer '{layer_name}' not found in model from client {first_client_id}. Skipping this layer for aggregation.")
                continue
                
            aggregated_layer_tensor = torch.zeros_like(example_tensor)
            aggregated_aux_tensor = torch.zeros_like(example_tensor)
            # Aggregate the layer from all clients in this round
            for i, client_idx in enumerate(clients_this_round):
                if client_idx not in local_model_dicts:
                    self.log.warning(f"Model dictionary for client {client_idx} not found. Skipping this client for layer {layer_name}.")
                    continue
                
                client_model_dict = local_model_dicts[client_idx]
                
                if layer_name in client_model_dict:
                    # Ensure client's layer tensor is on the target device before aggregation
                    client_layer_tensor = client_model_dict[layer_name].to(self.device)
                    aggregated_layer_tensor += client_layer_tensor * client_weights[i]
                else:
                    self.log.warning(f"Layer '{layer_name}' not found in model from client {client_idx}. Skipping this layer for this client.")
                aggregated_aux_tensor +=  self.local_upd[client_idx][layer_name] / len(list(self.sample_num_dict.keys()))
                self.local_c[client_idx][layer_name].to("cpu")
                self.local_upd[client_idx][layer_name].to("cpu")

            self.global_c[layer_name] += aggregated_aux_tensor
            global_dict[layer_name] = aggregated_layer_tensor

        self.global_state = global_dict

        self.log.info(f"Global aggregation complete for {len(global_dict)} layers.")
        return global_dict