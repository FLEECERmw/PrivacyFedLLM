import torch
from collections import defaultdict # Not strictly needed here but often useful
from typing import Dict, List, Any, Optional, Union
from .basepolicy import FedPolicy
import logging

module_logger = logging.getLogger(__name__)

class FedAvgPolicy(FedPolicy):
    def __init__(
        self,
        sample_num_dict: Dict[Any, Union[int, float]],
        logger: Optional[logging.Logger] = None,
        target_device: Optional[torch.device] = None,
        **kwargs 
    ):
        self.log = logger if logger is not None else module_logger
        self.sample_num_dict = sample_num_dict
        if target_device is not None:
            self.device = target_device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info(f"FedAvgPolicy initialized. Aggregation will run on device: {self.device}")
    
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
             
            global_dict[layer_name] = aggregated_layer_tensor

        self.log.info(f"Global aggregation complete for {len(global_dict)} layers.")
        return global_dict