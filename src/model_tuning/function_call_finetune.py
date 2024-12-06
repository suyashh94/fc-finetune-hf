import os
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format, SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import argparse


class ModelTrainer:
    def __init__(self, args):
        self.args = args
        self.IGNORE_INDEX = -100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.dataset_tokenized = None
        self.trainer = None

    def setup_environment(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    def initialize_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.base_model, trust_remote_code=True
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        self.model, self.tokenizer = setup_chat_format(self.model, self.tokenizer)

    def apply_lora(self):
        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'dense', 'fc1', 'fc2'
            ],
            modules_to_save=["lm_head", "embed_tokens"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.config.use_cache = False

    def tokenize(self, input):
        max_length = 1024
        input_ids, attention_mask, labels = [], [], []
        messages = [
            input['messages_templated']['system'],
            input['messages_templated']['user'],
            input['messages_templated']['assistant']
        ]
        for i, msg in enumerate(messages):
            msg_tokenized = self.tokenizer(msg, truncation=False, add_special_tokens=False)
            input_ids += msg_tokenized["input_ids"]
            attention_mask += msg_tokenized["attention_mask"]
            if i == 2:
                labels += msg_tokenized["input_ids"]
            else:
                labels += [self.IGNORE_INDEX] * len(msg_tokenized["input_ids"])
        return {
            "input_ids": input_ids[:max_length],
            "attention_mask": attention_mask[:max_length],
            "labels": labels[:max_length],
        }

    def collate(self, elements):
        tokens = [e["input_ids"] for e in elements]
        tokens_maxlen = max(len(t) for t in tokens)

        for i, sample in enumerate(elements):
            pad_len = tokens_maxlen - len(sample["input_ids"])
            sample["input_ids"].extend(pad_len * [self.tokenizer.pad_token_id])
            sample["labels"].extend(pad_len * [self.IGNORE_INDEX])
            sample["attention_mask"].extend(pad_len * [0])

        batch = {
            "input_ids": torch.tensor([e["input_ids"] for e in elements]),
            "labels": torch.tensor([e["labels"] for e in elements]),
            "attention_mask": torch.tensor([e["attention_mask"] for e in elements]),
        }

        return batch

    def prepare_data(self):
        complete_messages = np.load(self.args.complete_data_path, allow_pickle=True)
        if self.args.incomplete_data_path:
            incomplete_messages = np.load(self.args.incomplete_data_path, allow_pickle=True)
            num_samples = int(len(complete_messages) * 0.5)
            if len(incomplete_messages) < num_samples:
                num_samples = len(incomplete_messages)
                
            incomplete_messages = np.random.choice(incomplete_messages, num_samples, replace=False)
            messages = np.concatenate([complete_messages, incomplete_messages])
        else:
            messages = complete_messages

        dataset = Dataset.from_dict({"messages_templated": messages})
        dataset = dataset.shuffle(seed=42).train_test_split(test_size=0.1, seed=42)
        self.dataset_tokenized = dataset.map(
            self.tokenize,
            batched=False,
            num_proc=min(4, os.cpu_count()),
            remove_columns=dataset["train"].column_names
        )

    def configure_trainer(self):
        sft_config = SFTConfig(
            output_dir="/mnt",
            num_train_epochs=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=2e-5,
            max_seq_length=2048,
            dataset_text_field="messages_templated",
            save_strategy="epoch",
            logging_steps=50,
            eval_steps=25000,
            lr_scheduler_type="constant",
            eval_strategy="steps",
            report_to=[]
        )
        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset_tokenized["train"],
            eval_dataset=self.dataset_tokenized["test"],
            tokenizer=self.tokenizer,
            args=sft_config,
            data_collator=self.collate,
        )

    def train_model(self):
        self.trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.args.save_adapter_path)
        self.tokenizer.save_pretrained(self.args.save_adapter_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--complete_data_path", type=str, required=True)
    argparser.add_argument("--incomplete_data_path", type=str, default=None)
    argparser.add_argument("--save_adapter_path", type=str, default="../models/phi-2-adapter")
    argparser.add_argument("--base_model", type=str, default="microsoft/phi-2")
    args = argparser.parse_args()

    trainer = ModelTrainer(args)
    trainer.setup_environment()
    trainer.initialize_model_and_tokenizer()
    trainer.apply_lora()
    trainer.prepare_data()
    trainer.configure_trainer()
    trainer.train_model()
    trainer.save_model()