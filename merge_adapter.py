import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    """
    一个用于将PEFT LoRA适配器与基础模型融合并保存的脚本。
    """
    parser = argparse.ArgumentParser(description="将LoRA适配器与基础模型融合。")
    parser.add_argument(
        "--base_model_path", 
        type=str, 
        required=True, 
        help="基础语言模型（例如 Llama-2-7B）的路径或Hugging Face Hub名称。"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="包含 adapter_config.json 和 adapter_model.safetensors 的LoRA适配器检查点(ckpt)的目录路径。"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="保存融合后新模型的目录路径。"
    )
    args = parser.parse_args()

    print(f"开始加载基础模型: {args.base_model_path}")
    # 加载基础模型和分词器
    # 我们使用 bfloat16 来获得更好的性能和显存效率
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)

    print(f"加载LoRA适配器: {args.adapter_path}")
    # 加载PEFT模型，这将把LoRA模块加载到基础模型之上
    # 此时，模型是 PeftModel 类型
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    print("适配器加载完成。")

    print("开始融合适配器权重...")
    # 调用 merge_and_unload 将适配器权重合并到基础模型中
    # 执行后，模型会变回 AutoModelForCausalLM 类型
    model = model.merge_and_unload()
    print("权重融合完成。")

    print(f"保存融合后的完整模型到: {args.output_path}")
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 保存模型和分词器
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    
    print("所有操作完成！融合后的模型已保存。")

if __name__ == "__main__":
    main()