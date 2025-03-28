import json
import difflib
import tqdm
import fire
import os
import re
import hashlib
import glob
import subprocess

from typing import Optional, List
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
from pathlib import Path

script_dir = os.path.dirname(__file__)

def compute_diff(
    buggy_code: str, fixed_code: str, context_len: Optional[int] = None
) -> List[str]:
    """
    Computes the diff between the buggy and fixed code.
    """
    context_len = (
        context_len
        if context_len is not None
        else max(len(buggy_code), len(fixed_code))
    )
    with open("/tmp/buggy.java","w") as f: f.write(buggy_code+"\n")
    with open("/tmp/fixed_code.java","w") as f: f.write(fixed_code+"\n")
    # we want to ignore whitespace changes with -w which does not exist in difflib.unified_diff
    # with git diff, we even the name of the changed function in the diff, which helps a lot
    cmd = ["git","diff","--patience","-U10", "-w","/tmp/buggy.java","/tmp/fixed_code.java"]
    return subprocess.run(cmd, capture_output=True).stdout


def remove_java_comments(source: str) -> str:
    # Define states
    NORMAL, SINGLE_COMMENT, MULTI_COMMENT, STRING_LITERAL, CHAR_LITERAL = range(5)

    state = NORMAL
    result = []
    i = 0

    while i < len(source):
        # Check the current state and process accordingly
        if state == NORMAL:
            if source[i : i + 2] == "//":
                state = SINGLE_COMMENT
                i += 2
            elif source[i : i + 2] == "/*":
                state = MULTI_COMMENT
                i += 2
            elif source[i] == '"':
                state = STRING_LITERAL
                result.append(source[i])
                i += 1
            elif source[i] == "'":
                state = CHAR_LITERAL
                result.append(source[i])
                i += 1
            else:
                result.append(source[i])
                i += 1
        elif state == SINGLE_COMMENT:
            if source[i] == "\n":
                state = NORMAL
                result.append(source[i])
                i += 1
            else:
                i += 1
        elif state == MULTI_COMMENT:
            if source[i : i + 2] == "*/":
                state = NORMAL
                i += 2
            else:
                i += 1
        elif state == STRING_LITERAL:
            if source[i] == "\\":
                result.append(source[i])
                i += 1
                result.append(source[i])
                i += 1
            elif source[i] == '"':
                state = NORMAL
                result.append(source[i])
                i += 1
            else:
                result.append(source[i])
                i += 1
        elif state == CHAR_LITERAL:
            if source[i] == "\\":
                result.append(source[i])
                i += 1
                result.append(source[i])
                i += 1
            elif source[i] == "'":
                state = NORMAL
                result.append(source[i])
                i += 1
            else:
                result.append(source[i])
                i += 1

    return "".join(result)


def remove_empty_lines(source):
    """Remove all empty lines from Java source code."""
    return re.sub(r"^\s*$\n", "", source, flags=re.MULTILINE)


def manual_analysis_file(input_file: str, output_file: str, cache_path: str):
    """
    Perform manual analysis on the bugs in the input file and write the results to the output file.

    Args:
    - input_file: str: The path to the input file containing the bugs.
    - output_file: str: The path to the output file to write the results.
    - cache_path: str: The patch to the cache.
    """
    # Compute output file and clean it
    if os.path.exists(output_file):
        os.remove(output_file)

    # Load fixed functions
    fixed_functions = {}
    with open(f"{script_dir}/defects4j-fixed_code.jsonl") as f:
        # one bug per line
        for line in f.readlines():
            bug = json.loads(line)
            fixed_functions[bug["identifier"]] = bug["fixed_code"]
    with open(f"{script_dir}/humanevaljava-fixed_code.jsonl") as f:
        # one bug per line
        for line in f.readlines():
            bug = json.loads(line)
            fixed_functions[bug["identifier"]] = bug["fixed_code"]
    with open(f"{script_dir}/gitbugjava-fixed_code.jsonl") as f:
        # one bug per line
        for line in f.readlines():
            bug = json.loads(line)
            fixed_functions[bug["identifier"]] = bug["fixed_code"]

    # Read all bugs from the input_file
    bugs = []
    with open(input_file) as f:
        for line in f.readlines():
            bug = json.loads(line)
            bugs.append(bug)

    total_to_analyze = 0
    # Perform manual analysis
    for bug in tqdm.tqdm(bugs, "Number of bugs processed: "):
        try:
            # Sanity check
            if "identifier" not in bug:
                with open("problems.log","a") as f: f.write("a bug has no identifier in "+input_file+"\n")
                continue
            if "evaluation" not in bug:
                with open("problems.log","a") as f: f.write(bug["identifier"]+" has no evaluation\n")
                continue

            # Skip bugs that are not considered in our subset:
            if bug["identifier"] not in fixed_functions or fixed_functions[bug["identifier"]] is None:
                continue

            # Skip bugs not evaluated
            if bug["evaluation"] == None:
                continue

            # Skip bugs that are already semantical matches
            if any(x["exact_match"] or x["ast_match"] or ("semantical_match" in x and x["semantical_match"] == True) for x in bug["evaluation"] if x is not None):
                continue

            # Skip bugs that do not have plausible patches
            if not any(x["test"] for x in bug["evaluation"] if x is not None):
                continue

            for evaluation in bug["evaluation"]:
                # Skip if the patch is not plausible
                if not evaluation["test"]:
                    continue
                
                # Skip if semantical match is not disagree
                if evaluation["semantical_match"] != "Disagree":
                    continue
                
                patch = evaluation["generation"]
                cachename = Path(cache_path, hashlib.sha256(patch.encode()).hexdigest())
             
                result = 0
                # If we have already analyzed this patch, we load the result
                if os.path.exists(cachename):
                    with open(cachename) as f:
                        result = int(json.loads(f.read())["result"])
                        # total_to_analyze += 1
                # Otherwise we ask the user
                else:
                    total_to_analyze += 1

                    print("\033[1mYou are looking at:", bug["identifier"], "\033[0m")

                    # Print ground truth and plausible patch side-by-side in the terminal
                    ground_truth = compute_diff(
                        remove_empty_lines(remove_java_comments(bug["buggy_code"].strip())),
                        remove_empty_lines(
                            remove_java_comments(fixed_functions[bug["identifier"]].strip())
                        ),
                    )
                    plausible_patch = compute_diff(
                        remove_empty_lines(remove_java_comments(bug["buggy_code"].strip())),
                        remove_empty_lines(remove_java_comments(patch.strip())),
                    )

                    print("Ground truth:")
                    print(
                        highlight(
                            ground_truth, get_lexer_by_name("diff"), TerminalFormatter()
                        )
                    )
                    print("Plausible patch:")
                    print(
                        highlight(
                            plausible_patch, get_lexer_by_name("diff"), TerminalFormatter()
                        )
                    )

                    # Ask the user for the result
                    print("0 -> different, 1 -> equivalent")
                    while result != "0" and result != "1":
                        result = input("Option: ")
                    result = int(result)

                    # We store the result in the cache
                    with open(cachename,"w") as f:
                        f.write(json.dumps({"identifier":bug["identifier"],"patch":patch,"result":result}))
                # Update the bug with the result
                # Skip remainder of the patches if the user found a semantical match
                if result == 1:
                    evaluation["semantical_match"] = True
                    break
                else:
                    evaluation["semantical_match"] = False

                # Clear the terminal
                os.system("cls" if os.name == "nt" else "clear")
        finally:
            # Write the bug to the output file
            with open(output_file, "a+") as f:
                f.write(json.dumps(bug) + "\n")
    
    return total_to_analyze

def manual_analysis():
    results_path = f"{script_dir}/../../results/3_martin"
    cache_path = f"{script_dir}/../../results/3_martin/cache"

    for input_file in Path(f"{script_dir}/../../results/2_merged").glob("*.jsonl"):
        output_file = Path(results_path, input_file.name.replace("_merged.jsonl", f"_martin.jsonl"))
        total_to_analyze = manual_analysis_file(input_file, output_file, cache_path)

    print(total_to_analyze)


if __name__ == "__main__":
    fire.Fire(manual_analysis)