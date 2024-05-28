import json
import fire
import re
import gzip

def convert_files(*input_files):
    """
    Converts multiple jsonl files from Sen's format to elle-elle-aime's format
    """
    for input_file in input_files:
        convert_format(input_file)

def convert_format(input_file):
    """
    Connverts jsonl file from Sen's format to elle-elle-aime's format
    """
    # Read all bugs from the input_file
    bugs = []
    with open(input_file) as f:
        for line in f.readlines():
            bug = json.loads(line)
            bugs.append(bug)

    # Find string in format "irX_orX" and extract from input_file
    pattern = r"ir\d+_or\d+"
    ir_or = re.search(pattern, input_file)
    if ir_or:
        ir_or = ir_or.group()

    # Compute output_file
    input_dir = "/".join(input_file.split("/")[:-1])
    input_file = input_file.split("/")[-1]
    model_name = input_file.split("_")[0].lower()
    dataset_name = input_file.split("_")[1].lower()
    dataset_name = {"defects4j": "defects4j", "humaneval": "humanevaljava"}[dataset_name]    
    prompt_strategy = {"gpt4": "gpt-zero-shot", "gpt35": "gpt-zero-shot", "repairllama": ir_or}[model_name]
    suffix = "_andre" if "andre" in input_file else "_sen" if "sen" in input_file else "_merged" if "merged" in input_file else ""
    output_file = f"{input_dir}/evaluation_{dataset_name}_{model_name}_{prompt_strategy}{suffix}.jsonl"

    # Read fixed_functions
    fixed_functions = {}
    with open("defects4j-fixed_code.jsonl") as f:
        # one bug per line
        for line in f.readlines():
            bug = json.loads(line)
            fixed_functions[bug["identifier"]] = bug["fixed_code"]
    with open("humanevaljava-fixed_code.jsonl") as f:
        # one bug per line
        for line in f.readlines():
            bug = json.loads(line)
            fixed_functions[bug["identifier"]] = bug["fixed_code"]

    # Get buggy code
    buggy_functions = {}
    with open("/home/andre/Repos/repairllama/results/0_original/evaluation_defects4j_zero-shot-cloze_repairllama-fft.jsonl.gz", "rb") as gzfp:
        with gzip.open(gzfp, "rt") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    buggy_functions[json.loads(line)["identifier"]] = json.loads(line)["buggy_code"]

    # Sort according to the bug id
    bugs = sorted(bugs, key=lambda x: x["bug_id"])

    # Step 1. Replace "bug_id" key with "identifier"
    for bug in bugs:
        bug["buggy_code"] = buggy_functions[bug["bug_id"]]
        bug["identifier"] = bug.pop("bug_id")

    # Step 2. Keep "buggy_code" add "fixed_code"
    for bug in bugs:
        bug["fixed_code"] = fixed_functions[bug["identifier"]]

    # Step 3. Add "prompt_strategy", "prompt", and "ground_truth"
    for bug in bugs:
        bug["prompt_strategy"] = prompt_strategy
        bug["prompt"] = "TBA"
        bug["ground_truth"] = None

    # Step 4. Replace "patches" with "generation"
    for bug in bugs:
        bug["generation"] = bug.pop("patches")

    # Step 5. Replace "test_results" with "evaluation"
    for bug in bugs:
        bug["evaluation"] = [{} for _ in range(len(bug["generation"]))]
        test_results = bug.pop("test_results")
        for i, patch in enumerate(bug["generation"]):
            bug["evaluation"][i]["generation"] = patch
            bug["evaluation"][i]["exact_match"] = False
            bug["evaluation"][i]["ast_match"] = False
            bug["evaluation"][i]["semantical_match"] = False
            bug["evaluation"][i]["compile"] = False
            bug["evaluation"][i]["test"] = False

            if test_results[i].strip().lower() in ["Line Match", "Line match", "Match", "line match", "match"]:
                bug["evaluation"][i]["exact_match"] = True
                bug["evaluation"][i]["ast_match"] = True
                bug["evaluation"][i]["semantical_match"] = True
                bug["evaluation"][i]["compile"] = True
                bug["evaluation"][i]["test"] = True
            elif test_results[i].strip().lower() in ["AST Match", "AST match", "ast match"]:
                bug["evaluation"][i]["ast_match"] = True
                bug["evaluation"][i]["semantical_match"] = True
                bug["evaluation"][i]["compile"] = True
                bug["evaluation"][i]["test"] = True
            elif test_results[i].strip().lower() in ["Semantical Match", "Semantical match", "semantical match"]:
                bug["evaluation"][i]["semantical_match"] = True
                bug["evaluation"][i]["compile"] = True
                bug["evaluation"][i]["test"] = True
            elif test_results[i].strip().lower() in ["Disagree", "disagree"]:
                bug["evaluation"][i]["semantical_match"] = "Disagree"
                bug["evaluation"][i]["compile"] = True
                bug["evaluation"][i]["test"] = True
            elif test_results[i].strip().lower() in ["Plausible", "plausible"]:
                bug["evaluation"][i]["semantical_match"] = False
                bug["evaluation"][i]["compile"] = True
                bug["evaluation"][i]["test"] = True
            elif test_results[i].strip().lower() in ["Test Fail", "Test fail", "test fail"]:
                bug["evaluation"][i]["test"] = False
                bug["evaluation"][i]["compile"] = True
            elif test_results[i].strip().lower() in ["Compile Fail", "Compile fail", "compile fail"]:
                bug["evaluation"][i]["compile"] = False
            else:
                print("Unknown test result:", test_results[i])
                print("Bug id:", bug["identifier"])

    # Output the converted bugs
    with open(output_file, "w") as f:
        for bug in bugs:
            f.write(json.dumps(bug) + "\n")

if __name__ == "__main__":
    fire.Fire(convert_files)
