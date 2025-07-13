
# FedASK: Differentially Private Federated Low-Rank Adaptation
This repository contains the official implementation for the paper: **"Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix"**.

## ✨ Key Features

This framework supports a variety of flexible experimental configurations out of the box:

* **📊 Data Distribution**: Simulate both **IID** and **Non-IID** data distributions using a Dirichlet process to model data heterogeneity.
* **🔒 Privacy Settings**: Seamlessly switch between standard non-private training and **Differentially Private (DP)** training, powered by the Opacus library.
* **💻 Training Modes**: Supports both **single-GPU** and **multi-GPU** training on a single machine, leveraging PyTorch's `DistributedDataParallel` (DDP) for efficient scaling.
* **🧭 Federated Policies**: Includes a suite of built-in federated learning policies such as `FedAvg`, `FedProx`, `SCAFFOLD`, and our proposed `FedASK`, which can be easily configured.

## 🛠️ Setup

1.  **Clone the Repository**

2.  **Create and Activate Conda Environment**
    We recommend using Conda to manage the project's dependencies.
    ```bash
    conda create -n pfedllm python=3.10
    conda activate pfedllm
    ```

3.  **Install Dependencies**
    First, install PyTorch, ensuring it matches your system's CUDA version. This project was tested with `torch==2.4.1` and CUDA 12.1.
    ```bash
    # Example for CUDA 12.1
    pip install torch torchvision torchaudio
    ```
    Next, install the remaining required packages:
    ```bash
    pip install -i requirements.txt
    ```

## 🚀 Running the Experiment

The framework uses a centralized `config.yaml` file to manage all experimental settings and a simple `sft.sh` script to launch the training.

**1. Modify the Configuration File (`config.yaml`)**

Before launching, configure your experiment by editing `config.yaml`.

* **Models & Paths**:
    * `name_model`: Set the base model to use (e.g., `"NousResearch/Llama-2-7b-hf"`).
    * `data.name_localdatadir`: Specify a local path for your dataset to avoid re-downloading.
    * `checkpoint_dir`: Set the directory where model checkpoints will be saved.
* **Federated & Training Settings**:
    * `policy.name`: Switch between federated policies (e.g., `"fedask"`, `"fedavg"`).
    * `data.mode_datadist`: Set the data distribution mode (`"iid"` or `"dirichlet-noniid"`).
    * `privacy.use_privacy`: Set to `true` to enable differentially private training.
* **Distributed Training**:
    * Modify the `PROC_PER_NODE` variable in `sft.sh` to set the number of GPUs to use.

**2. Launch Training**

Once the configuration is set, execute the launch script:
```bash
bash sft.sh
````

This script leverages `torchrun` to initiate the training process according to the settings defined in `config.yaml`.

**3. Merge LoRA Adapter (Optional)**

After training, the saved checkpoints contain only the LoRA adapter weights. To obtain a full, standalone model, merge the adapter with the base model using the `merge_adapter.py` script.

```bash
python merge_adapter.py \
    --base_model_path [path_to_your_base_model] \
    --adapter_path [path_to_your_lora_checkpoint] \
    --output_path [path_to_save_the_merged_model]
```

## Citation

If you find our work useful in your research, please consider citing our paper:

<!-- ```bibtex
@article{anonymous2025fedask,
  title={Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix},
  author={Anonymous Author(s)},
  journal={Submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025)},
  year={2025}
} -->
```
