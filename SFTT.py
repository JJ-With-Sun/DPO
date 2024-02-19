from transformers import (
  AutoModelForCausalLM, 
  AutoTokenizer, 
  BitsAndBytesConfig,
#   HfArgumentParser,
  TrainingArguments,
  pipeline, 
  logging, 
  TextStreamer,
  )
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, wandb, platform, warnings,sys
from datasets import load_dataset, Dataset
from trl import SFTTrainer,DataCollatorForCompletionOnlyLM
import argparse
import torch
import numpy as np
import random

from sklearn.model_selection import train_test_split
# from huggingface_hub import notebook_login


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",type=str,required=True)
    parser.add_argument("--dataset_path",type=str,required=True)
    parser.add_argument("--wandb_key",type=str)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--dropout_rate", type=float, default=0.05)
    parser.add_argument("--max_len", type=int, default=2048)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epoch_size', type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=2)
    parser.add_argument("--gradient_checkpointing",type=bool,default=False,help="reduce required memory size but slower training")
    parser.add_argument("--ckpt_path",type=str,default=None)
    parser.add_argument("--train",action="store_true")
    # parser.add_argument("--test",action="store_true")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--project_desc",type=str, default = "Fine tuning llm")
    parser.add_argument("--name",type=str,default=None, help="file name to add")
    parser.add_argument("--full_ft",action="store_true",help="full finetuning otherwise lora")
    # parser.add_argument("--lora_target_modules",)
    # parser.add_argument("--quantize")

    return parser.parse_args()


def create_prompt(instruction_template,instruction,response_template,response,input_template,system_definition=None,input_text=None):
  
  """
  example)
  system_definition : "Below is an instruction that describes a task. Write a response that appropriately completes the request."
  instruction_template : "[INST]"
  input_template : "### input:"
  response_template: "[/INST]"
  """

  if input_text is not None:
    prompt= """{instruction_template}{instruction}    

{input_template}{input_text} {response_template} {response}""".format(
            system_definition=system_definition,
            instruction_template=instruction_template,
            instruction=instruction,
            response_template=response_template,
            response=response,
            input_template=input_template,
            input_text=input_text,
        )

  else:
    prompt= """{instruction_template}{instruction} {response_template} {response}""".format(
        system_definition=system_definition,
        instruction_template=instruction_template,
        instruction=instruction,
        response_template=response_template,
        response=response,
    )
    
    
  if system_definition is not None:
    prompt=system_definition+prompt
    

  return prompt

def load_and_prepare_dataset(dataset_path,system_definition,instruction_template,response_template,input_template):
  # load a raw dataset from huggingface and reconstruct it in the defined prompt form 

  dataset=Dataset.from_json(dataset_path)
  def _add_text(sample):
    # add text field(column)

    # input과 instrurction이 반대인 데이터셋
    instruction=sample['input']
    context=sample.get('instruction') # template can be 'context' or others
    response=sample["output"] # template can be 'response' or others
    
    if context:
      sample['text']=create_prompt(system_definition=system_definition,
      instruction_template=instruction_template,
      instruction=instruction,
      response_template=response_template,
      response=response,
      input_template=input_template,
      input_text=context,
      )
    else:
      sample["text"]=create_prompt(system_definition=system_definition,
      instruction_template=instruction_template, 
      instruction=instruction,
      response_template=response_template,
      response=response,
      input_template=input_template,
      )
    return sample
  return dataset.map(_add_text)


def tokenize_dataset(dataset,tokenizer,max_length):
    # this fucntion guarantees every training samples must have complete input texts
      
  dataset=dataset.filter(lambda x: len(tokenizer.tokenize(x["text"]))<max_length-1) # should consider bos/eos token
  dataset=dataset.map(lambda x: tokenizer(x['text'],max_length=max_length), batched=True)
  dataset=dataset.shuffle()
  return dataset
  
def load_tokenizer(base_model_path,additional_special_tokens:list[str]=None):
  """
  base_model_path : path or name of pretrained model
  additional_special_tokens : additional special tokens to add ex) ['### instruction:', 'input:']
  """
  tokenizer = AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  
  if additional_special_tokens is not None:
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )
  tokenizer.padding_side="right"
  return tokenizer


def load_model(base_model_path, gradient_checkpointing=False,quantization_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        trust_remote_code=True, 
        use_cache=False if gradient_checkpointing else True, # use_cache is incompatible with gradient_checkpointing
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )
    return model

def main(args):

    ## for quantization
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit= True,
    #     bnb_4bit_quant_type= "nf4",
    #     bnb_4bit_compute_dtype= torch.float16,
    #     bnb_4bit_use_double_quant= False,
    # )

    system_definition=None
    instruction_template="[INST]"
    response_template="[/INST]"
    input_template="질문 : "
    
    model=load_model(args.base_model,gradient_checkpointing=args.gradient_checkpointing,quantization_config=None)
    tokenizer=load_tokenizer(args.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False # use_cache is only for infernce
    # model.config.pretraining_tp = 1 


    ## dataset
    dataset=load_and_prepare_dataset(args.dataset_path,system_definition,instruction_template,response_template,input_template)
    dataset=tokenize_dataset(dataset,tokenizer,args.max_len)
    dataset=dataset.train_test_split(test_size=0.1)
    train_dataset=dataset["train"]
    eval_dataset=dataset["test"]
    
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer.encode(response_template,add_special_tokens=False),tokenizer=tokenizer)

    # wandb
    wandb.login(key = args.wandb_key)
    run = wandb.init(project=args.project_desc, job_type="training", anonymous="allow")

    ## q lora
    # model=prepare_model_for_kbit_training(model)



    if not args.full_ft:
      ## peft (lora)
      peft_config = LoraConfig(
              r=16,
              lora_alpha=16,
              lora_dropout=args.dropout_rate,
              bias="none",
              task_type="CAUSAL_LM",
              target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","down_proj","up_proj"]
          )
      
      model=get_peft_model(model,peft_config)


    if args.name is not None:
      output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}_{args.name}"
    else:
      output_dir= f"./checkpoints/{args.base_model.split('/')[-1]}"
    
    training_arguments = TrainingArguments(
        output_dir= output_dir,
        num_train_epochs= args.epoch_size,
        per_device_train_batch_size= args.batch_size,
        per_device_eval_batch_size= args.batch_size,
        gradient_accumulation_steps= args.gradient_accumulation_steps,
        learning_rate= args.learning_rate,
        weight_decay= args.weight_decay,
        optim = "paged_adamw_32bit",
        evaluation_strategy="steps",
        save_steps= 10,
        logging_steps= 100,
        eval_steps=10,
        save_total_limit=5,
        # save_strategy="epoch"
        fp16= False,
        # bf16= True,
        # max_grad_norm= 0.3,
        # max_steps= -1,
        warmup_ratio= 0.1,
        # group_by_length= True,
        lr_scheduler_type= "linear",
        report_to="wandb"
    )

    trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config if not args.full_ft else None,
    max_seq_length= args.max_len,
    dataset_text_field="text",
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_arguments,
    # neftune_noise_alpha=5,
    # packing= True,
)


    if args.train:
      
      if args.ckpt_path is not None:
        trainer.train(args.ckpt_path)
      else:
        trainer.train()
      wandb.finish()

# if args.test:


if __name__=="__main__":
  seed_everything()
  args=parse_args()
  main(args)