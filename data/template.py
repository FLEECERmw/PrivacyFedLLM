


alpaca_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{} 

### Response: {}{}"""


kpmath_template = """User: Please reason step by step and put your final answer at the end with "The answer is: ".

{}

Assistant:
{}{}"""


TEMPLATE_DICT = {
    'alpaca': (alpaca_template, '\n### Response:'),
    'kpmath': (kpmath_template,'\n\nAssistant:\n'),
}


def get_formatting_prompts_func(template_name, eos_token):
    overall_temp, response_temp = TEMPLATE_DICT[template_name]
    def formatting_prompts_func(example):    
        output_texts = []    
        for i in range(len(example['instruction'])):    
            text = overall_temp.format(example['instruction'][i], example['response'][i], eos_token)    
            output_texts.append(text)  
        return output_texts    
    
    return formatting_prompts_func, response_temp
    


# test

# from transformers import AutoModelForCausalLM, AutoTokenizer
# alpaca_test = '\n### Response:'
# kpmath_test = '\n\nAssistant:\n'

# model_name = "/data/fdu_model/hub/models--NousResearch--Llama-2-7b-hf/snapshots/8efe6c9b93655b934e27bd9981e3ec13e55aee9d"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, padding_side="right")
# encoded_full = tokenizer.encode(kpmath_test, add_special_tokens=False)

# text = kpmath_template.format("test this is a.", "also a test.", tokenizer.eos_token )
# test_full = tokenizer.encode(text, add_special_tokens=False)

# [29871, 13, 2277, 29937, 13291, 29901]
# [29871, 13, 13, 7900, 22137, 29901, 13]