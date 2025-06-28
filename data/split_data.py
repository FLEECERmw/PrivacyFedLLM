import numpy as np
from collections import defaultdict, Counter
from datasets import Dataset
from typing import List, Optional, Union
import random
import logging

module_logger = logging.getLogger(__name__)

def split_dataset(
    dataset: Dataset,
    num_clients: int = 1,
    split_mode: str = 'iid',
    noniid_degree: Optional[float] = None,
    seed: str = '42',
    logger: Optional[logging.Logger] = None
) -> List[Dataset]:
    """
    Splits a Hugging Face Dataset for federated learning simulation.
    'iid' mode distributes types as evenly as possible without a specific per-type cap.
    In 'dirichlet-noniid' according to a dirichlet distribution with non iid degree, smaller non-iid degreee represents a severe heterogeneity.
    """
    log = logger if logger is not None else module_logger
    type_column: str = 'type'

    if not isinstance(dataset, Dataset):
        log.error("Input 'dataset' must be a Dataset object.")
        raise TypeError("Input 'dataset' must be a Dataset object.")
    if num_clients <= 0:
        log.error("num_clients must be positive.")
        raise ValueError("num_clients must be positive.")
    if split_mode not in ['iid', 'dirichlet-noniid']:
        log.error(f"Invalid split_mode: {split_mode}. Must be 'iid' or 'dirichlet-noniid'.")
        raise ValueError("split_mode must be 'iid' or 'dirichlet-noniid'.")

    try:
        current_seed = int(seed)
    except ValueError:
        log.warning(f"Could not convert seed '{seed}' to an integer. Using default seed 42.")
        current_seed = 42

    np.random.seed(current_seed)
    random.seed(current_seed)

    log.info(f"Grouping dataset indices by '{type_column}'...")
    type_data = defaultdict(list)
    try:
        for idx, example in enumerate(dataset):
            type_value = example[type_column]
            type_data[type_value].append(idx)
    except KeyError:
        log.error(f"Column '{type_column}' not found in dataset features: {dataset.features}.")
        raise ValueError(f"Column '{type_column}' not found in dataset features: {dataset.features}.")

    available_types = sorted(list(type_data.keys()))

    if not available_types:
        log.warning(f"No data for '{type_column}' or dataset empty. Returning empty client datasets.")
        return [dataset.select([]) for _ in range(num_clients)]

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    if split_mode == 'iid':
        log.info(f"Splitting dataset in 'iid' mode for {num_clients} clients.")
        for type_key in available_types:
            type_specific_indices = type_data[type_key][:]
            num_samples_of_type = len(type_specific_indices)
            if num_samples_of_type == 0: continue
            np.random.shuffle(type_specific_indices)
            num_to_take_per_client = num_samples_of_type // num_clients

            if num_to_take_per_client == 0:
                if num_samples_of_type > 0:
                    log.warning(f"Type '{type_key}' (total {num_samples_of_type}) has insufficient samples for even distribution "
                                f"to {num_clients} clients (0 per client). This type will not be distributed this way.")
                continue
            for i in range(num_clients):
                start_idx = i * num_to_take_per_client
                end_idx = start_idx + num_to_take_per_client
                client_indices[i].extend(type_specific_indices[start_idx:end_idx])

    elif split_mode == 'dirichlet-noniid':
        log.info(f"Splitting dataset in 'dirichlet-noniid' mode for {num_clients} clients (alpha={noniid_degree}).")
        if noniid_degree is None or noniid_degree <= 0:
            log.error("noniid_degree (alpha) must be a positive float for 'dirichlet-noniid' mode.")
            raise ValueError("noniid_degree (alpha) must be a positive float for 'dirichlet-noniid' mode.")

        # 新增：追踪每个客户端的样本数，用于实现均衡
        client_sample_counts = np.zeros(num_clients, dtype=int)

        for type_key in available_types:
            type_specific_indices = type_data[type_key][:]
            num_samples_of_type = len(type_specific_indices)
            if num_samples_of_type == 0: continue
            np.random.shuffle(type_specific_indices)

            if num_samples_of_type < num_clients:
                 log.warning(f"Type '{type_key}' ({num_samples_of_type} samples) < num_clients ({num_clients}). "
                             f"Some clients get 0 samples of this type in Dirichlet split.")

            # --- 为实现均衡而做的核心修改 ---
            # 目标：让当前样本量较少的客户端，有更高概率获得新类型的数据。
            # 我们通过调整狄利克雷分布的alpha参数来实现这一点。
            # alpha_i 越大，客户端i能分到的比例的期望就越大。
            
            total_samples_so_far = np.sum(client_sample_counts)
            
            # 如果已经分配过数据，则计算平衡权重
            if total_samples_so_far > 0:
                # 客户端当前数据量越小，其权重越高。
                # +1e-8 是为了防止除以零。
                ideal_size_per_client = total_samples_so_far / num_clients
                balance_weights = ideal_size_per_client / (client_sample_counts + 1e-8)
                
                # 将权重和基础的noniid_degree结合，生成新的alphas向量
                # 这样，样本少的客户端会有更大的alpha值
                alphas_for_dirichlet = noniid_degree * balance_weights
            else:
                # 对于第一个数据类型，所有客户端的alpha相同
                alphas_for_dirichlet = [noniid_degree] * num_clients

            proportions = np.random.dirichlet(alphas_for_dirichlet)
            # --- 修改结束 ---

            if num_samples_of_type > 0 :
                # 沿用原有的、稳健的np.split逻辑来切分数据
                split_points = (np.cumsum(proportions)[:-1] * num_samples_of_type).astype(int)
                try:
                    client_type_indices_list = np.split(type_specific_indices, split_points)
                    for i in range(num_clients):
                        if i < len(client_type_indices_list):
                            indices_to_add = client_type_indices_list[i]
                            client_indices[i].extend(list(indices_to_add))
                            # 更新客户端的样本计数
                            client_sample_counts[i] += len(indices_to_add)
                except ValueError as e:
                    # 原有的错误处理逻辑保持不变
                    log.warning(f"Error with np.split for type '{type_key}': {e}. Attempting manual assignment.")
                    current_pos = 0
                    for i in range(num_clients):
                        num_to_take_for_client = int(round(proportions[i] * num_samples_of_type)) if i < num_clients -1 else num_samples_of_type - current_pos
                        actual_take = max(0, min(num_to_take_for_client, num_samples_of_type - current_pos))
                        
                        indices_to_add = type_specific_indices[current_pos : current_pos + actual_take]
                        client_indices[i].extend(indices_to_add)
                        client_sample_counts[i] += len(indices_to_add)
                        current_pos += actual_take

    local_datasets: List[Dataset] = []
    log.info("Creating client datasets...")
    for i in range(num_clients):
        client_specific_indices = client_indices[i]
        if not client_specific_indices:
            log.warning(f"Client {i} has no data assigned.")
        np.random.shuffle(client_specific_indices)
        local_datasets.append(dataset.select(client_specific_indices))
        log.info(f"Client {i} dataset: {len(client_specific_indices)} samples. Types: {dict(Counter(local_datasets[-1][type_column])) if client_specific_indices else 'N/A'}")

    return local_datasets


# test
if __name__ == '__main__':
    from load_data import get_dataset, sample_dataset
    from collections import Counter

    
    logging.basicConfig(
            level=logging.INFO, # Change to DEBUG to see more details if functions add debug logs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    # dataset = get_dataset("meta-math/MetaMathQA", local_data_dir="/data/fdu_model/hub/datasets--meta-math--MetaMathQA/snapshots/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
    dataset = get_dataset("databricks/databricks-dolly-15k", local_data_dir="/data/fdu_model/hub/datasets--databricks--databricks-dolly-15k/snapshots/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
    dataset = sample_dataset(dataset, 15000)   

    local_client_datasets = split_dataset(dataset, 10, split_mode='dirichlet-noniid', noniid_degree=0.1)

    breakpoint()
