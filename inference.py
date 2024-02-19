import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from transformers import TextStreamer, GenerationConfig
from datasets import load_dataset, Dataset

import warnings
warnings.filterwarnings('ignore')

base_model_name='davidkim205/komt-mistral-7b-v1'
peft_model_name = "/home/kic/yskids/inbound/inbound_model/checkpoints/komt-mistral-7b-v1_lora_sftt/checkpoint-180"

base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cuda:0",torch_dtype=torch.float16)

peft_model = PeftModel.from_pretrained(base_model, peft_model_name)
tokenizer = AutoTokenizer.from_pretrained(peft_model_name)

peft_model=peft_model.merge_and_unload()

dataset_path="/home/kic/yskids/inbound/inbound_model/data/instruction_tuning_test.jsonl"
dataset=Dataset.from_json(dataset_path)

prompt= "[INPUT]" + dataset['input'][0] + "[INST]" + dataset['instruction'][0] + "[/INST]"

generation_config = GenerationConfig(
        temperature=0.1,
        # top_p=0.8,
        # top_k=100,
        max_new_tokens=512,
        repetiton_penalty=1.2,
        early_stopping=True,
        do_sample=True,
    )

gened = peft_model.generate(
        **tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        #streamer=streamer,
    )
result_str = tokenizer.decode(gened[0])
start_tag = f"[/INST]"
start_index = result_str.find(start_tag)

if start_index != -1:
    result_str = result_str[start_index + len(start_tag):].strip()
print(result_str)
