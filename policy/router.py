import logging
from typing import Dict, Any, Type
from policy import FedASKPolicy, FedAvgPolicy, FedProxPolicy, ScaffoldPolicy

module_logger = logging.getLogger(__name__) 

POLICY_REGISTRY = {
    "fedask": FedASKPolicy,
    "fedavg": FedAvgPolicy,
    "fedffa": FedAvgPolicy,
    "scaffold": ScaffoldPolicy,
    "fedprox": FedProxPolicy,
}

def get_policy(policy_name: str, policy_config: Dict[str, Any]) -> Any:
    """
    Instantiates and returns a policy object based on its name and configuration.

    Args:
        policy_name (str): The name of the policy (e.g., "fedask", "fedavg").
        policy_config (Dict[str, Any]): A dictionary containing parameters
                                         required for the policy's initialization.

    Returns:
        Any: An instantiated policy object.

    Raises:
        ValueError: If the provided policy_name is invalid or not found.
        TypeError: If parameters in policy_config do not match the
                   __init__ method of the selected policy class.
    """
    policy_name_lower = policy_name.lower() # For case-insensitive matching
    policy_class = POLICY_REGISTRY.get(policy_name_lower)

    if policy_class is None:
        raise ValueError(
            f"Unknown Policy name: '{policy_name}'. "
            f"Available Policies: {list(POLICY_REGISTRY.keys())}"
        )

    try:
        # Unpack policy_config as keyword arguments to the constructor
        return policy_class(**policy_config)
    except TypeError as e:
        # Catches errors if policy_config arguments don't match __init__ signature
        raise TypeError(
            f"Error initializing policy '{policy_name}'. "
            f"Please check if policy_config keys match parameters for {policy_class.__name__}.__init__. "
            f"Original error: {e}"
        )
    except Exception as e:
        # Catches other potential initialization errors
        module_logger.error(f"An unexpected error occurred while initializing policy '{policy_name}': {e}", exc_info=True)
        raise