from datasets import load_dataset
import json

def transform_input(example):
    input_text = example['input']
    input_text = input_text.replace("<FILL_ME>", "<｜fim▁hole｜>")
    input_text = f"<｜fim▁begin｜>{input_text}<｜fim▁end｜>"
    example['input'] = input_text
    return example

if __name__ == "__main__":
    dataset = load_dataset("ASSERT-KTH/repairllama-datasets", "ir4xor2")
    
    # Transform the dataset
    dataset["train"] = dataset["train"].map(transform_input)
    dataset["test"] = dataset["test"].map(transform_input)
    
    # Save to jsonl files
    dataset["train"].to_json("train.jsonl")
    dataset["test"].to_json("test.jsonl")