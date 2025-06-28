import torch
from collections import defaultdict # Not strictly needed here but often useful
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from .basepolicy import FedPolicy

module_logger = logging.getLogger(__name__)

class FedProxPolicy(FedPolicy):
    def __init__(
        self,
        sample_num_dict: Dict[Any, Union[int, float]],
        global_state: Dict[Any, torch.Tensor],
        logger: Optional[logging.Logger] = None,
        target_device: Optional[torch.device] = None,
        prox_mu: float = 0.01,
        **kwargs 
    ):
        self.log = logger if logger is not None else module_logger
        self.sample_num_dict = sample_num_dict
        self.global_state = global_state
        self.prox_mu = prox_mu
        if target_device is not None:
            self.device = target_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"FedProxpolicy initialized. Aggregation will run on device: {self.device}")
    
    def local_compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        return_values = super(FedProxPolicy, self).local_compute_loss(model, inputs, return_outputs)
        prefix = ["_module", "module"]
        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            else:
                name = name.replace(".default", "").replace('_module.', '', 1).replace('module.', '', 1)
                loss += self.prox_mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss  
    
    def global_aggregate(
        self,
        local_model_dicts: Dict[Any, Dict[str, torch.Tensor]], # Key: client_id
        clients_this_round: List[Any] # List of client_ids participating
    ) -> Dict[str, torch.Tensor]:

        if not clients_this_round:
            self.log.warning("No clients provided for aggregation. Returning empty dictionary.")
            return {}

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
        
        client_weights = client_sample_counts / total_samples_this_round
        
        global_dict: Dict[str, torch.Tensor] = {}
        
        if not local_model_dicts or clients_this_round[0] not in local_model_dicts:
            self.log.error("local_model_dicts is empty or first client in clients_this_round not found in local_model_dicts.")
            return {}
            
        first_client_id = clients_this_round[0]
        layer_names = local_model_dicts[first_client_id].keys()

        for layer_name in layer_names:
            try:
                example_tensor = local_model_dicts[first_client_id][layer_name].to(self.device)
            except KeyError:
                self.log.warning(f"Layer '{layer_name}' not found in model from client {first_client_id}. Skipping this layer for aggregation.")
                continue
                
            aggregated_layer_tensor = torch.zeros_like(example_tensor)
            
            for i, client_idx in enumerate(clients_this_round):
                if client_idx not in local_model_dicts:
                    self.log.warning(f"Model dictionary for client {client_idx} not found. Skipping this client for layer {layer_name}.")
                    continue
                
                client_model_dict = local_model_dicts[client_idx]
                
                if layer_name in client_model_dict:
                    client_layer_tensor = client_model_dict[layer_name].to(self.device)
                    aggregated_layer_tensor += client_layer_tensor * client_weights[i]
                else:
                    self.log.warning(f"Layer '{layer_name}' not found in model from client {client_idx}. Skipping this layer for this client.")
            
            global_dict[layer_name] = aggregated_layer_tensor
        
        self.global_state = global_dict # update global_state

        self.log.info(f"Global aggregation complete for {len(global_dict)} layers.")
        return global_dict