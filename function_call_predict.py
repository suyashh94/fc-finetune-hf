import torch
from peft import PeftConfig, PeftModel, PeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import json

# --------------------------------------------------------------------------------------
# *** Load model for inference

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_save_path", type=str, default="./phi-2-adapter")
argparser.add_argument("--device_name", type=str, default="cuda")
argparser.add_argument("--mode", type=str, default="test")
argparser.add_argument("--eval_file", type=str, default="none")

args = argparser.parse_args()
model_save_path = args.model_save_path
device_name = args.device_name
mode = args.mode
eval_file = args.eval_file

if mode == "eval":
    assert eval_file != "none", "Please provide the evaluation file path."

device = torch.device(device_name)
config = PeftConfig.from_pretrained(model_save_path)
base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
  )

#### CHANGE THIS IF YOU ARE USING ANYTHING OTHER THAN PHI-2 #### 
#### NUMBERS WILL CHANGE, RUN IT ONCE, LOOK AT THE ERROR AND CHANGE IT ACCORDINGLY ####

new_vocab_size = 50297  # This should match the size from the checkpoint error
original_embedding_dim = base_model.model.embed_tokens.weight.shape[1]

# Resize the embeddings
base_model.model.embed_tokens.weight.data = base_model.model.embed_tokens.weight.data[:new_vocab_size]
base_model.lm_head.weight.data = base_model.lm_head.weight.data[:new_vocab_size]
base_model.lm_head.bias.data = base_model.lm_head.bias.data[:new_vocab_size]

############# LOAD MODEL #############
finetuned_model = PeftModel.from_pretrained(
    base_model, model_save_path
)

########### LOAD TOKENIZER ############
loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)

if mode != "eval":
    
    while True:
        user_query = input("Enter your query: \n")
        messages = [
            {
                "role":"system", 
                # "content":"You are a helpful assistant with access to the following functions. Use these functions when they are relevant to assist with a user's request\n[\n	{\n		\"name\": \"calculate_retirement_savings\",\n		\"description\": \"Project the savings at retirement based on current contributions.\",\n		\"parameters\": {\n			\"type\": \"object\",\n			\"properties\": {\n				\"current_age\": {\n					\"type\": \"integer\",\n					\"description\": \"The current age of the individual.\"\n				},\n				\"retirement_age\": {\n					\type\": \"integer\",\n					\"description\": \"The desired retirement age.\"\n				},\n				\"current_savings\": {\n					\"type\": \"number\",\n					\"description\": \"The current amount of savings.\"\n				},\n				\"monthly_contribution\": {\n					\"type\": \"number\",\n					\"description\": \"The monthly contribution towards retirement savings.\"\n				}\n			},\n			\"required\": [\"current_age\", \"retirement_age\", \"current_savings\", \"monthly_contribution\"]\n		}\n	}\n]"
                "content": "You are a helpful assistant. You have to either provide a way to answer user's request or answer user's query."
            },
            {
                "role": "user", 
                "content": user_query
            }
        ]
        
        input_text = loaded_tokenizer.apply_chat_template(messages,tokenize=False)
        input_ids = loaded_tokenizer(input_text, return_tensors="pt").to('cuda')

        outputs = finetuned_model.to('cuda').generate(**input_ids, max_length=128)
        print("Model response: ", loaded_tokenizer.decode(outputs[0])[len(input_text):])

else:
    with open(eval_file) as f:
        eval_data = json.load(f)
    
    #shuffle
    import random
    random.shuffle(eval_data)
    
    n = 100
    
    for i,data in enumerate(eval_data):
        system_message = data["system"]
        user_message = data["user"]
        input_text = system_message + user_message
        input_ids = loaded_tokenizer(input_text, return_tensors="pt").to('cuda')
        
        outputs = finetuned_model.to('cuda').generate(**input_ids, max_length=128)
        data["model_response"] = loaded_tokenizer.decode(outputs[0])[len(input_text):]
        
        if i > n:
            break
    
    # import pdb; pdb.set_trace()
    
    fname = './data/predicted_outputs.json'
    with open(fname, "w") as f:
        json.dump(eval_data[:n+2], f)
        
        
        
    
