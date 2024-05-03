# coding=utf-8
# Implements parameter-efficient or full parameters supervised fine-tuning for LLaMa model.
# This code is inspired by
# https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py and https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning


import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    Seq2SeqTrainer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

import torch
import os
import evaluate
import functools
from datasets import load_dataset
# import bitsandbytes as bnb
import logging
import json
import copy
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field


# Lora settings
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="TheBloke/CodeLlama-7B-fp16")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    train_file: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    eval_file: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    cache_path: str = field(default=None, metadata={"help": "Path to the cache directory."})
    num_proc: int = field(default=4, metadata={"help": "Number of processes to use for data preprocessing."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # adam_beta1: float = field(default=0.9)
    # adam_beta2: float = field(default=0.95)
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    is_lora: bool = field(default=True, metadata={"help": "Whether to use LORA."})


def tokenize(text, tokenizer, max_seq_len=1024, add_eos_token=True):
    result = tokenizer(
        text,
        truncation=False,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
    )

    # If the tokenized length exceeds the max_seq_len, return None
    if len(result["input_ids"]) >= max_seq_len:
        return None

    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < max_seq_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    # if add_eos_token and len(result["input_ids"]) >= max_seq_len:
    #     result["input_ids"][max_seq_len - 1] = tokenizer.eos_token_id
    #     result["attention_mask"][max_seq_len - 1] = 1

    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.is_lora:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=data_args.cache_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_8bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            ),
        )
        model = prepare_model_for_kbit_training(model)

        config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            cache_dir=data_args.cache_path,
            trust_remote_code=True,
        )
    model.config.use_cache = False


    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    if training_args.is_lora:
        print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=data_args.cache_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load dataset

    def generate_and_tokenize_prompt(sample):

        input_text = sample["input"]
        target_text = sample["output"] + tokenizer.eos_token
        full_text = input_text + target_text

        tokenized_full_text = tokenize(full_text, tokenizer, max_seq_len=training_args.model_max_length)

        if tokenized_full_text is None:
            # Return a null sample if the tokenized length exceeds the max_seq_len
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized_input_text = tokenize(input_text, tokenizer, max_seq_len=training_args.model_max_length)
        input_len = len(tokenized_input_text["input_ids"]) # This a bug of llamatokenizer that it does not add eos token
        tokenized_full_text["labels"] = [-100] * input_len + tokenized_full_text["labels"][input_len:]
        return tokenized_full_text

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.eval_file is not None:
        data_files["eval"] = data_args.eval_file

    dataset = load_dataset(data_args.data_path, data_files=data_files)
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    def print_dataset_length(dataset, name):
        print(f"Number of samples in {name} dataset after filtering: {len(dataset)}")


    train_dataset = train_dataset.map(generate_and_tokenize_prompt, num_proc=data_args.num_proc)
    eval_dataset = eval_dataset.map(generate_and_tokenize_prompt, num_proc=data_args.num_proc)
    # Filter null samples
    train_dataset = train_dataset.filter(lambda sample: len(sample["input_ids"]) > 0)
    eval_dataset = eval_dataset.filter(lambda sample: len(sample["input_ids"]) > 0)

    print_dataset_length(train_dataset, "train")
    print_dataset_length(eval_dataset, "eval")

    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    # Evaluation metrics
    def compute_metrics(eval_preds, tokenizer):
        metric = evaluate.load('exact_match')
        preds, labels = eval_preds
        # In case the model returns more than the prediction logits
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Replace -100s in the labels as we can't decode them
        labels[labels == -100] = tokenizer.pad_token_id
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {'exact_match': result['exact_match']} 

    compute_metrics_fn = functools.partial(compute_metrics, tokenizer=tokenizer)

    # Training
    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    tokenizer.save_pretrained(save_directory=training_args.output_dir)


if __name__ == "__main__":
    main()

