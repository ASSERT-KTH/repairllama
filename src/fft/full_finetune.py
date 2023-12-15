import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import evaluate
import tqdm
import numpy
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from functools import partial

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="codellama/CodeLlama-7b-hf")
    torch_dtype: torch.dtype = field(default=torch.bfloat16)
    device_map: str = field(default="auto")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="XXX")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=1024)
    # bf16: bool = field(default=True)
    # bf16_full_eval: bool = field(default=True)
    num_train_epochs: float = field(default=2.0)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    learning_rate: float = field(default=2e-5)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_epsilon: float = field(default=10e-5)
    weight_decay: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.02)
    logging_steps: int = field(default=1)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=200)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=1000)
    report_to: str = field(default="wandb")
    seed: int = field(default=42)

def tokenize(text, tokenizer, training_args, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=True,
        max_length=training_args.model_max_length,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < training_args.model_max_length
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    if add_eos_token and len(result["input_ids"]) >= training_args.model_max_length:
        result["input_ids"][training_args.model_max_length - 1] = tokenizer.eos_token_id
        result["attention_mask"][training_args.model_max_length - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def get_prompt_target(sample):
    return sample['input'], sample['output']

def generate_and_tokenize_prompt(sample, tokenizer, training_args):
    input_text, target = get_prompt_target(sample)
    full_text = input_text + target + tokenizer.eos_token
    tokenized_full_text = tokenize(full_text, tokenizer, training_args)
    tokenized_input_text = tokenize(input_text, tokenizer, training_args)
    input_len = len(tokenized_input_text["input_ids"]) # This a bug of llamatokenizer that it does not add eos token
    tokenized_full_text["labels"] = [-100] * input_len + tokenized_full_text["labels"][input_len:]
    return tokenized_full_text

def get_data_module(tokenizer, training_args) -> Dict:
    ds = load_dataset("./ior_21")

    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    train_dataset = train_dataset.map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args))
    eval_dataset = eval_dataset.map(partial(generate_and_tokenize_prompt, tokenizer=tokenizer, training_args=training_args))
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    print(model_args)
    print(training_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        torch_dtype=model_args.torch_dtype,
        device_map=model_args.device_map,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    data_module = get_data_module(tokenizer=tokenizer, training_args=training_args)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
