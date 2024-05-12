import json
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers
from peft import PeftModel
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    GenerationConfig, 
    HfArgumentParser, 
    BitsAndBytesConfig,
)
from tqdm import tqdm


@dataclass
class ModelArguments:
    base_model_path: Optional[str] = field(default="TheBloke/CodeLlama-7B-fp16")
    lora_path: Optional[str] = field(default="../repairllama-lora")
    max_length: int = field(default=512, metadata={"help": "Maximum length of the input sequence."})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    test_file: str = field(default="test.json", metadata={"help": "Test file name."})
    output_file: str = field(default=None, metadata={"help": "Output file name."})


@dataclass
class GenerationArguments:
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    is_lora: bool = field(default=True, metadata={"help": "Whether to use LORA."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to use sampling."})
    only_do_beam: bool = field(default=False, metadata={"help": "Whether to only use beam search."})
    only_do_topp: bool = field(default=False, metadata={"help": "Whether to only use top-p sampling."})
    only_do_topk: bool = field(default=False, metadata={"help": "Whether to only use top-k sampling."})
    only_do_temp: bool = field(default=False, metadata={"help": "Whether to only use temperature sampling."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search. 1 means no beam search."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature for sampling."})
    top_k: int = field(default=50, metadata={"help": "Top-k for sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Top-p for sampling."})
    request_num: int = field(default=1, metadata={"help": "Number of requests."})
    sub_request_num: int = field(default=10, metadata={"help": "Number of requests for each bug."})


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments))
    model_args, data_args, generation_args = parser.parse_args_into_dataclasses()

    if generation_args.is_lora:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.base_model_path,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            ),
        )

        model = PeftModel.from_pretrained(
            model,
            model_args.lora_path,
            torch_dtype=torch.float16,
        )

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.base_model_path,
            torch_dtype=torch.float16,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model_path, trust_remote_code=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.config.pad_token = tokenizer.pad_token = tokenizer.unk_token
    model.to(device)
    
    if generation_args.do_sample:
        if generation_args.only_do_beam:
            # Beam search
            print("We now use beam search to generate the patches.")
            generation_config = GenerationConfig(
                num_beams=generation_args.num_beams,
                early_stopping=True,
            )
        else:
            # The combination of Top-k & Top-p & Temperature sampling
            print("We now use sampling strategies to generate the patches.")
            generation_config = GenerationConfig(
                do_sample=generation_args.do_sample,
                temperature=generation_args.temperature if generation_args.only_do_temp else None,
                top_k=generation_args.top_k if generation_args.only_do_topk else None,
                top_p=generation_args.top_p if generation_args.only_do_topp else None,
            )

    buggy_code_list = []
    llama2_output = []
    data_path = Path(data_args.data_path)
    with open(data_path / data_args.test_file, 'r') as tf:
        for sample in tf.readlines():
            buggy_code_list.append(json.loads(sample))
    
    # If we want to generate 100 patches for a bug, we need to generate 10 times due to the limited GPU resources.
    if generation_args.only_do_temp or generation_args.only_do_topk or generation_args.only_do_topp:
        gen_epoch = int(generation_args.request_num / generation_args.sub_request_num)

    for sample in tqdm(buggy_code_list, desc="Generating..."):
        tmp_dict = {}
        buggy_code = sample["buggy_code"]

        inputs = tokenizer(buggy_code, return_tensors='pt')
        inputs_len = inputs.input_ids.shape[1]
        input_ids = inputs.input_ids.to(device)

        if generation_args.only_do_beam:
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=generation_args.max_new_tokens,
                        num_return_sequences=generation_args.request_num,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        generation_config=generation_config,
                    )
            except:
                print("The code sequence of bug {} is too long, {}.".format(sample['bug_id'], inputs_len))
                continue
            output_ids = outputs[:, inputs_len:]

            output_diff = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            original_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            output_dict = {}

            for i in range(len(output_diff)):
                output_dict[i] = {
                    "original_output": original_outputs[i],
                    "output_patch": output_diff[i],
                }
            
            tmp_dict['bug_id'] = sample['bug_id']
            tmp_dict['output'] = output_dict
            tmp_dict['buggy_code'] = buggy_code
            tmp_dict['gold_patch'] = sample['fixed_chunk']

            llama2_output.append(tmp_dict)
        else:
            temp_list = []
            for _ in range(gen_epoch):
                try:
                    with torch.no_grad():
                        outputs = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=generation_args.max_new_tokens,
                            num_return_sequences=generation_args.sub_request_num,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            generation_config=generation_config,
                        )
                except:
                    print("The code sequence of bug {} is too long, {}.".format(sample['bug_id'], inputs_len))
                    break
                output_ids = outputs[:, inputs_len:]

                output_diff = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                original_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                temp_list.append((original_outputs, output_diff))

            output_dict = {}

            for i in range(len(temp_list)):
                for j in range(len(temp_list[i][1])):
                    output_dict[i*10+j] = {
                        "original_output": temp_list[i][0][j],
                        "output_diff": temp_list[i][1][j],
                    }
            
            tmp_dict['bug_id'] = sample['bug_id']
            tmp_dict['output'] = output_dict

            if len(output_dict) == generation_args.request_num:
                llama2_output.append(tmp_dict)

    with open(data_args.output_file, 'w') as pd_file:
        for each in llama2_output:
            pd_file.write(json.dumps(each))
            pd_file.write('\n')


if __name__ == "__main__":
    main()

