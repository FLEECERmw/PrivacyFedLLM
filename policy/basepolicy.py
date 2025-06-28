import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union # For type hints
import logging
from abc import ABC, abstractmethod

module_logger = logging.getLogger(__name__)

class FedPolicy(ABC):
    """
    Base class for Federated Learning policies.

    This class defines the interface for different federated aggregation strategies
    and local computation steps. Subclasses should implement the specific logic
    for `global_aggregate` and potentially override other methods.
    """
    def __init__(self, logger: Optional[logging.Logger] = None, **kwargs):
        """
        Initializes the base federated policy.

        Args:
            logger: Optional logger instance. If None, uses module_logger.
            **kwargs: Placeholder for any additional arguments future policies might need.
        """
        self.log = logger if logger is not None else module_logger
        # You might want to process common kwargs here if any are expected by all policies
    
    @abstractmethod
    def global_aggregate(
        self,
        local_model_dicts: Dict[Any, Dict[str, torch.Tensor]],
        clients_this_round: List[Any]
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregates model updates from multiple clients to update the global model.
        This method **must be implemented by subclasses**.

        Args:
            local_model_dicts: A dictionary mapping client IDs to their local model
                               state dictionaries (torch.nn.Module.state_dict()).
            clients_this_round: A list of client IDs that participated in this round.

        Returns:
            A dictionary representing the updated global model's state dictionary.
        """
        pass

    def local_compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Computes the loss for a given model and inputs on a client.
        This implementation is similar to how Hugging Face Trainer's compute_loss works
        at its core for causal language models or sequence classification.

        Args:
            model: The model (e.g., a PyTorch nn.Module) to compute the loss for.
            inputs: A dictionary of input tensors to the model.
                    It's expected to contain 'input_ids', 'attention_mask', and 'labels'.
            return_outputs (bool): Whether to return model outputs along with the loss.

        Returns:
            If return_outputs is False, returns only the loss tensor.
            If return_outputs is True, returns a tuple (loss, model_outputs).
        
        Raises:
            ValueError: If 'labels' are not found in inputs, as they are required for loss computation.
        """
        if "labels" not in inputs:
            self.log.error("Labels not found in inputs. Cannot compute loss.")
            raise ValueError("`labels` must be provided in inputs to compute loss.")

        try:
            device = next(model.parameters()).device
            inputs_on_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        except StopIteration: # Model has no parameters
            self.log.warning("Model has no parameters. Assuming inputs are already on the correct device.")
            inputs_on_device = inputs
        except Exception as e:
            self.log.warning(f"Could not determine model device or move inputs: {e}. Using inputs as is.")
            inputs_on_device = inputs

        outputs = model(**inputs_on_device)

        if hasattr(outputs, "loss") and outputs.loss is not None:
            loss = outputs.loss
        elif isinstance(outputs, tuple) and len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
            loss = outputs[0]
        else:
            self.log.error(
                "The model did not return a 'loss' attribute or a tuple where the first element is the loss. "
                "Please ensure your model's forward method computes and returns loss when 'labels' are provided."
            )
            raise ValueError(
                "Could not retrieve loss from model outputs. "
                "Ensure model's forward pass returns loss when 'labels' are provided."
            )

        return (loss, outputs) if return_outputs else loss