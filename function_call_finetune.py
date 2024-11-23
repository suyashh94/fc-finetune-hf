import os
import numpy as np
# import wandb
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format
from peft import LoraConfig, get_peft_model, cast_mixed_precision_params, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer
import math
from dotenv import load_dotenv
from transformers import GenerationConfig

import os
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_path", type=str, default="./data/car_finetuning_False.npy")
argparser.add_argument("--save_adapter_path", type=str, default="./phi-2-adapter")
argparser.add_argument("--base_model", type=str, default="microsoft/phi-2")

args = argparser.parse_args()
data_path = args.data_path
save_adapter_path = args.save_adapter_path
base_model = args.base_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

############# MODEL PART #############
base_model_path = base_model
# base_model_path = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(base_model_path)

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model_path,
    trust_remote_code=True,
  )

model.config.use_cache = False # disable the use of the cache during the 
                               # generation process.
model.config.pretraining_tp = 1 # disables tensor parallelism, the model is
                                # not split across multiple devices and runs
                                # on a single device.     
# Set up the chat format with default 'chatml' format
model, tokenizer = setup_chat_format(model, tokenizer)

############ LORA PART ############
# LoRA Configuration
peft_config = LoraConfig(
    r=32, 
    lora_alpha=32,
    target_modules=[
        'q_proj',
        'k_proj',
        'v_proj',
        'dense',
        'fc1',
        'fc2',
    ], 
    modules_to_save = ["lm_head", "embed_tokens"],
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

# Apply LoRA to the Model:
model = get_peft_model(model, peft_config)
model.config.use_cache = False 

########### DATA PART ############
IGNORE_INDEX = -100

def tokenize(input):
    max_length = 1024 
    input_ids, attention_mask, labels = [], [], [] 
    message = [input['messages_templated']['system'],
               input['messages_templated']['user'],
               input['messages_templated']['assistant']]
   
    for i, msg in enumerate(message):
        msg_tokenized = tokenizer(  
          msg,   
          truncation=False,   
          add_special_tokens=False)  
  
        # Copy tokens and attention mask without changes  
        input_ids += msg_tokenized["input_ids"]  
        attention_mask += msg_tokenized["attention_mask"]
        
        # Adapt labels for loss calculation: if system or user ->IGNORE_INDEX, 
        # if assistant->input_ids  (calculate loss only for assistant messages)      
        if i == 2:
            labels += msg_tokenized["input_ids"]  
        else:
            labels += [IGNORE_INDEX]*len(msg_tokenized["input_ids"]) 
    
    # truncate to max. length  
    return {  
        "input_ids": input_ids[:max_length],   
        "attention_mask": attention_mask[:max_length],  
        "labels": labels[:max_length],  
    }  

def collate(elements):
    tokens=[e["input_ids"] for e in elements]
    tokens_maxlen=max([len(t) for t in tokens])

    for i,sample in enumerate(elements):
        input_ids=sample["input_ids"]
        labels=sample["labels"]
        attention_mask=sample["attention_mask"]

        pad_len=tokens_maxlen-len(input_ids)

        input_ids.extend( pad_len * [tokenizer.pad_token_id] )   
        labels.extend( pad_len * [IGNORE_INDEX] )    
        attention_mask.extend( pad_len * [0] ) 

    batch={
        "input_ids": torch.tensor( [e["input_ids"] for e in elements] ).to(model.device),
        "labels": torch.tensor( [e["labels"] for e in elements]).to(model.device),
        "attention_mask": torch.tensor( [e["attention_mask"] for e in elements] ).to(model.device),
    }

    return batch

messages = np.load(data_path, allow_pickle=True)

dataset = Dataset.from_dict({
    "messages_templated": messages 
    }
)

NUM_SAMPLES = len(dataset)

dataset = dataset.shuffle(seed=42).select(range(NUM_SAMPLES if NUM_SAMPLES < len(dataset) else len(dataset)))

dataset = dataset.train_test_split(test_size=0.1, seed=42)

dataset_tokenized = dataset.map(tokenize,   
            batched = False,  
            num_proc = os.cpu_count(),    # multithreaded  
            remove_columns = dataset["train"].column_names  # Remove original columns, no longer needed  
)


########### TRAINING PARAMS ############
max_seq_length = 2048 
batch_size = 2 
gradient_accum_steps = 1 
epochs = 1 
eval_steps = 50 
save_steps = eval_steps * 2 
logging_steps = 50 
lr = 2e-5   

print("Eval Steps:", eval_steps)
print("Save Steps:", save_steps)



new_model_name = "phi-2-function-calling-message-prompted-False"
new_model_path = f"./{new_model_name}"

train_model_name = f"{new_model_name}-train"
train_model_path = f"./{train_model_name}"


# Configure Training with SFTConfig:
sft_config = SFTConfig(
    output_dir=train_model_path, # Where the trained model will be saved.
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accum_steps,
    dataloader_pin_memory=False,
    optim="paged_adamw_32bit", 
    save_strategy="epoch",
    logging_steps=logging_steps,
    logging_strategy="steps",
    learning_rate=lr, 
    fp16=False,
    bf16=False,
    group_by_length=True,
    disable_tqdm=False,
    max_seq_length=max_seq_length,
    dataset_text_field="messages_templated", # Which field in the dataset contains the text data to be used.
    packing=False, 
    report_to=[],
    run_name=train_model_name,
    eval_steps=eval_steps,
    save_steps=save_steps,
    lr_scheduler_type="constant", 
    eval_strategy="steps",
  )

# Initialize the SFTTrainer:
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_tokenized['train'],
    eval_dataset=dataset_tokenized['test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=sft_config,
    data_collator=collate,
)

######## TRAIN ############

trainer.train()

############ SAMPLE PREDICTION ############
if "false" in data_path.lower():
    print("Manually add function to system message to get correct results like the one that is commented out below.")
    
messages = [
    {
        "role":"system", 
        # "content":"You are a helpful assistant with access to the following functions. Use these functions when they are relevant to assist with a user's request\n[\n	{\n		\"name\": \"calculate_retirement_savings\",\n		\"description\": \"Project the savings at retirement based on current contributions.\",\n		\"parameters\": {\n			\"type\": \"object\",\n			\"properties\": {\n				\"current_age\": {\n					\"type\": \"integer\",\n					\"description\": \"The current age of the individual.\"\n				},\n				\"retirement_age\": {\n					\type\": \"integer\",\n					\"description\": \"The desired retirement age.\"\n				},\n				\"current_savings\": {\n					\"type\": \"number\",\n					\"description\": \"The current amount of savings.\"\n				},\n				\"monthly_contribution\": {\n					\"type\": \"number\",\n					\"description\": \"The monthly contribution towards retirement savings.\"\n				}\n			},\n			\"required\": [\"current_age\", \"retirement_age\", \"current_savings\", \"monthly_contribution\"]\n		}\n	}\n]"
        "content": "You are a helpful assistant. You have to either provide a way to answer user's request or answer user's query."
    },
    {
        "role": "user", 
        "content": "Set temperature to 70 degrees."
    }
]


input_text = tokenizer.apply_chat_template(messages,tokenize=False)
input_ids = tokenizer(input_text, return_tensors="pt").to('cuda')

outputs = trainer.model.generate(**input_ids, max_length=128)
print("Input Text:", input_text)
print("Model output: ", tokenizer.decode(outputs[0]))

############## SAVE MODEL ##############
model_save_path = f"./phi-2-adapter"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
                         