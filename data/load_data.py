import os
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from collections import Counter
import numpy as np
import random
import logging

module_logger = logging.getLogger(__name__)

def get_dataset(
    dataset_name: str,
    local_data_dir: str = None,
    logger: logging.Logger = None
) -> Dataset:
    """
    Loads and preprocesses a specified dataset to ['instruction', 'response', 'type'].
    """
    log = logger if logger is not None else module_logger
    log.info(f"Processing dataset: {dataset_name}")
    path_or_name = local_data_dir if local_data_dir is not None else dataset_name

    processed_dataset = None
    try:
        if dataset_name == "databricks/databricks-dolly-15k":
            dataset = load_dataset(path_or_name, split="train")
            def dolly_format(example):
                example['type'] = example['category']
                return example
            processed_dataset = dataset.map(
                dolly_format,
                remove_columns=['category', 'context'],
                load_from_cache_file=False
            )
            processed_dataset = processed_dataset.select_columns(['instruction', 'response', 'type'])

        elif dataset_name == "meta-math/MetaMathQA":
            dataset = load_dataset(path_or_name, split="train", name='default')
            def metamath_kpmath_format(example):
                example['instruction'] = example['query']
                return example
            processed_dataset = dataset.map(
                metamath_kpmath_format,
                remove_columns=['query', 'original_question'],
                load_from_cache_file=False
            )
            processed_dataset = processed_dataset.select_columns(['instruction', 'response', 'type'])
        else:
            log.warning(f"No specific processing for '{dataset_name}'. Attempting generic load.")
            processed_dataset = load_dataset(path_or_name, split="train")
            if not all(col in processed_dataset.column_names for col in ['instruction', 'response', 'type']):
                log.warning(f"Generic dataset '{dataset_name}' may not have all required columns. "
                            f"Current: {processed_dataset.column_names}")
    except Exception as e:
        log.error(f"Error processing dataset {dataset_name}: {e}")
        raise
    
    log.info(f"Finished processing for {dataset_name}. Examples: {len(processed_dataset) if processed_dataset else 'Error'}.")
    return processed_dataset


def sample_dataset(
    dataset: Dataset,
    dataset_sample: int,
    seed: int = 42,
    type_weight: dict = None,
    logger: logging.Logger = None
) -> Dataset:
    """
    Stratified sampling from a Hugging Face Dataset.
    Optimized to reduce multiple filter operations.
    """
    log = logger if logger is not None else module_logger
    type_column = 'type'

    if not isinstance(dataset, Dataset):
        log.error("Input 'dataset' must be a Dataset object.")
        raise TypeError("Input 'dataset' must be a Dataset object.")
    if dataset_sample < 0:
        log.error("dataset_sample must be non-negative.")
        raise ValueError("dataset_sample must be non-negative.")
    if type_column not in dataset.column_names:
        log.error(f"Column '{type_column}' not found. Available: {dataset.column_names}")
        raise ValueError(f"Column '{type_column}' not found.")

    log.info("--- Initial Dataset Information for Sampling ---")
    initial_type_counts = Counter(dataset[type_column])
    log.info(f"Total examples before sampling: {len(dataset)}, Type distribution: {dict(initial_type_counts)}")
    log.info("-------------------------------------------------")

    actual_target_samples = min(len(dataset), dataset_sample)

    if actual_target_samples == 0 or len(dataset) == 0:
        log.info("Resulting dataset is empty (target or original dataset was empty).")
        return dataset.select(range(0)) if len(dataset) > 0 else Dataset.from_dict({col: [] for col in dataset.column_names})


    shuffled_dataset = dataset.shuffle(seed=seed)
    type_counts_shuffled = Counter(shuffled_dataset[type_column])
    unique_types = sorted(list(type_counts_shuffled.keys()))

    if not unique_types:
        log.warning(f"No unique types in '{type_column}'. Performing random sampling.")
        return shuffled_dataset.select(range(actual_target_samples))

    ideal_samples_per_type_float = {}
    effective_type_weights = type_weight if type_weight else {}

    if not effective_type_weights: # Uniform
        if unique_types:
            base_ideal = actual_target_samples / len(unique_types)
            for t in unique_types: ideal_samples_per_type_float[t] = base_ideal
    else: # Weighted
        active_weights = {t: effective_type_weights.get(t, 0) for t in unique_types if effective_type_weights.get(t, 0) > 0}
        total_weight = sum(active_weights.values())
        if total_weight == 0:
            log.warning("Total weight is zero for active types. Falling back to uniform sampling.")
            if unique_types:
                base_ideal = actual_target_samples / len(unique_types)
                for t in unique_types: ideal_samples_per_type_float[t] = base_ideal
        else:
            for t in unique_types:
                ideal_samples_per_type_float[t] = (active_weights.get(t, 0) / total_weight) * actual_target_samples

    current_samples_per_type = {}
    total_allocated_samples = 0
    for t in unique_types:
        ideal = ideal_samples_per_type_float.get(t, 0)
        available = type_counts_shuffled[t]
        num_to_take = min(int(ideal), available)
        current_samples_per_type[t] = num_to_take
        total_allocated_samples += num_to_take

    samples_still_needed = actual_target_samples - total_allocated_samples
    if samples_still_needed > 0:
        priority_list = []
        for t in unique_types:
            deserved = ideal_samples_per_type_float.get(t, 0) - current_samples_per_type[t]
            if deserved > 1e-6 and current_samples_per_type[t] < type_counts_shuffled[t]:
                priority_list.append((t, deserved))
        priority_list.sort(key=lambda x: x[1], reverse=True)

        for t, _ in priority_list:
            if samples_still_needed <= 0: break
            if current_samples_per_type[t] < type_counts_shuffled[t]:
                current_samples_per_type[t] += 1
                samples_still_needed -= 1

        if samples_still_needed > 0:
            eligible_types = [t for t in unique_types if current_samples_per_type[t] < type_counts_shuffled[t]]
            random.Random(seed + 1).shuffle(eligible_types)
            idx = 0
            safety_iter = 0
            max_safety_iter = len(unique_types) * actual_target_samples + 1
            while samples_still_needed > 0 and eligible_types and safety_iter < max_safety_iter:
                current_type = eligible_types[idx % len(eligible_types)]
                if current_samples_per_type[current_type] < type_counts_shuffled[current_type]:
                    current_samples_per_type[current_type] += 1
                    samples_still_needed -= 1
                else:
                    eligible_types = [t for t in eligible_types if current_samples_per_type[t] < type_counts_shuffled[t]]
                    if not eligible_types: break
                    idx = (idx -1) if len(eligible_types) > 0 else 0 # Adjust index carefully
                idx += 1
                safety_iter +=1
            if safety_iter >= max_safety_iter: log.warning("Remainder distribution safety break triggered.")

    indices_by_type = {t: [] for t in unique_types}
    for i, example_type_val in enumerate(shuffled_dataset[type_column]):
        if example_type_val in indices_by_type:
            indices_by_type[example_type_val].append(i)

    all_selected_indices = []
    final_actual_samples_count = 0
    for t in unique_types:
        num_to_sample_for_type = current_samples_per_type.get(t, 0)
        if num_to_sample_for_type > 0:
            selected_indices_for_this_type = indices_by_type[t][:num_to_sample_for_type]
            all_selected_indices.extend(selected_indices_for_this_type)
            final_actual_samples_count += len(selected_indices_for_this_type)

    if not all_selected_indices:
        final_sampled_dataset = dataset.select(range(0))
        try: final_sampled_dataset = final_sampled_dataset.cast(dataset.features)
        except Exception: pass
    else:
        final_sampled_dataset = shuffled_dataset.select(all_selected_indices)

    final_sampled_dataset_shuffled = final_sampled_dataset.shuffle(seed=seed + 2)

    log.info("--- Sampled Dataset Information ---")
    final_type_counts = Counter(final_sampled_dataset_shuffled[type_column])
    log.info(f"Target: {actual_target_samples}, Collected: {final_actual_samples_count}, Final count: {len(final_sampled_dataset_shuffled)}")
    log.info(f"Type distribution after sampling: {dict(final_type_counts) if final_type_counts else 'Empty'}")
    log.info("------------------------------------")

    if final_actual_samples_count != actual_target_samples and actual_target_samples > 0 and all_selected_indices:
        log.warning(f"Target samples {actual_target_samples}, but collected {final_actual_samples_count} due to constraints.")

    return final_sampled_dataset_shuffled



# test
if __name__ == '__main__':
    
    logging.basicConfig(
            level=logging.INFO, # Change to DEBUG to see more details if functions add debug logs
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    dataset = get_dataset("meta-math/MetaMathQA", local_data_dir="/data/fdu_model/hub/datasets--meta-math--MetaMathQA/snapshots/aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18")
    # dataset = get_dataset("databricks/databricks-dolly-15k", local_data_dir="/data/fdu_model/hub/datasets--databricks--databricks-dolly-15k/snapshots/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a")
    dataset = sample_dataset(dataset, 100000)   
