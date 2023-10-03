import fire
import subprocess
import json
import sys
import torch
import difflib
import uuid
import traceback

from pathlib import Path
from typing import Optional, List
from java_tools.java_lang import load_origin_code_node
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def generate_patch(
    statement: dict, dir_java_src: str, prompt: str
) -> Optional[List[str]]:
    # Step 1: load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        "CodeLlama-7b-hf",
        torch_dtype=torch.float16,
        #            device_map="auto",
        load_in_8bit=True,
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0
        ),
    )

    model = PeftModel.from_pretrained(
        model,
        "RepairLLaMa-Lora-7B-MegaDiff",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("RepairLLaMa-Lora-7B-MegaDiff")
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token = tokenizer.eos_token
    model.to(device)

    # Step 2: generate the output
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs_len = inputs["input_ids"].shape[1]
    input_ids = inputs["input_ids"].to(device)
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=64,
                num_beams=5,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except:
        traceback.print_exc()
        print("The code sequence is too long, {}.".format(inputs_len))
        return None

    output_ids = outputs[:, inputs_len:]
    outputs = tokenizer.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    outputs = [output.split("</s>")[0] for output in outputs]

    # Step 3: Compute the patch
    source_file = Path(
        dir_java_src, statement["className"].replace(".", "/") + ".java"
    ).absolute()
    with open(source_file, "r", encoding="ISO-8859-1") as file:
        buggy_code = file.readlines()

    diffs = []
    for output in outputs:
        fixed_code = buggy_code.copy()
        fixed_code[statement["lineNumber"] - 1] = f"{output}"

        diff = "".join(
            difflib.unified_diff(
                buggy_code,
                fixed_code,
                fromfile=str(source_file),
                tofile=str(source_file),
            )
        )

        diffs.append(diff)

    return diffs


def find_code(file_path: str, line_numbers: List[int]) -> str:
    """
    Finds the code corresponding to the given line numbers in the given file.
    """
    code = ""
    with open(file_path, "r", encoding="ISO-8859-1") as file:
        for idx, line in enumerate(file.readlines()):
            if idx + 1 in line_numbers:
                code += line
    return code


def generate_prompt(
    statement: dict,
    dir_java_src: str,
) -> Optional[str]:
    # Step 1: Compute the source file path
    source_file = Path(dir_java_src, statement["className"].replace(".", "/") + ".java")

    # Step 2: Find the function where the statement is located
    allowed_node_types = ["MethodDeclaration", "ConstructorDeclaration"]
    buggy_method = load_origin_code_node(
        source_file,
        [statement["lineNumber"]],
        allowed_node_types,
    )[0]

    # Step 3: Get the buggy code with line numbers
    # If the ast nodes are not of the correct type, then we have a whole-function removal/addition
    buggy_code = (
        find_code(
            source_file,
            [i for i in range(buggy_method.start_pos, buggy_method.end_pos + 1)],
        )
        if buggy_method is not None
        else ""
    ).strip()

    if buggy_code == "":
        return None

    # Step 4: Iterate over the buggy code to generate the prompt
    prompt = ""
    buggy_code = buggy_code.splitlines(keepends=True)
    for i, line in enumerate(range(buggy_method.start_pos, buggy_method.end_pos + 1)):
        if line == statement["lineNumber"]:
            prompt += f"// buggy lines start:\n"
            prompt += buggy_code[i]
            prompt += f"// buggy lines end\n"
        else:
            prompt += buggy_code[i]

    prompt += "// fixed lines:\n"

    return prompt


def run_flacoco(
    dir_java_src, dir_test_src, dir_java_bin, dir_test_bin
) -> Optional[List[dict]]:
    """
    Runs flacoco and returns the fault localization report.
    """
    command = (
        f"java -jar flacoco.jar "
        + f"--binJavaDir {dir_java_bin} "
        + f"--binTestDir {dir_test_bin} "
        + f"--srcJavaDir {dir_java_src} "
        + f"--srcTestDir {dir_test_src} "
        + f"--format JSON "
        + f"--output "
    )
    run = subprocess.run(command, shell=True)

    if run.returncode != 0:
        return None

    # Extract flacoco results (flacoco_results.json)
    with open("flacoco_results.json", "r") as f:
        flacoco_results = json.load(f)

    return flacoco_results


def main(dir_java_src, dir_test_src, dir_java_bin, dir_test_bin, patch_directory):
    # Step 1: Run Fault Localization with flacoco
    flacoco_results = run_flacoco(
        dir_java_src, dir_test_src, dir_java_bin, dir_test_bin
    )

    if flacoco_results is None:
        print("Error running flacoco")
        return -1

    # Step 2: Select the top K most suspicious statements
    K = 5
    suspicious_statements = flacoco_results[:5]

    # Step 3: Generate a patch for each suspicious statement
    patches = []
    for statement in suspicious_statements:
        prompt = generate_prompt(statement, dir_java_src)

        if prompt is None:
            continue

        new_patches = generate_patch(statement, dir_java_src, prompt)
        if new_patches is None:
            continue
        else:
            patches.extend(new_patches)

    # Step 4: Place the diffs in the correct directory
    for i, patch in enumerate(patches):
        Path(patch_directory).mkdir(parents=True, exist_ok=True)
        with open(Path(patch_directory, f"{i}.patch"), "w") as f:
            f.write(patch)


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
