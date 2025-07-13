import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    """
    A script to merge PEFT LoRA adapters with the base model and save the result.
    """
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter with a base model.")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True, 
        help="The path or Hugging Face Hub name of the base language model (e.g., Llama-2-7B)."
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="The directory path of the LoRA adapter checkpoint (ckpt) containing adapter_config.json and adapter_model.safetensors."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="The directory path to save the new, merged model."
    )
    args = parser.parse_args()

    print(f"Starting to load the base model: {args.base_model_path}")
    # Load the base model and tokenizer
    # We use bfloat16 for better performance and memory efficiency
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    print(f"Loading the LoRA adapter: {args.adapter_path}")
    # Load the PEFT model, which will load the LoRA modules on top of the base model
    # At this point, the model is of type PeftModel
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    print("Adapter loaded successfully.")

    print("Starting to merge adapter weights...")
    # Call merge_and_unload to combine the adapter weights into the base model
    # After execution, the model will revert to the AutoModelForCausalLM type
    model = model.merge_and_unload()
    print("Weight merging complete.")

    print(f"Saving the merged full model to: {args.output_path}")
    # Create the output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save the model and tokenizer
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("All operations complete! The merged model has been saved.")

if __name__ == "__main__":
    main()