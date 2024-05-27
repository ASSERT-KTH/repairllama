import json
import difflib
import hashlib
import tqdm
import fire
import gzip
import os
import subprocess
import re

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


def stream_jsonl(filename: str):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def manual_analysis_file(input_file: str, output_file: str, cache_path: str):
    """
    Perform manual analysis on the bugs in the input file and write the results to the output file.

    Args:
    - input_file: str: The path to the input file containing the bugs.
    - output_file: str: The path to the output file to write the results.
    """
    if output_file.exists():
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

    # Read all bugs from the input_file
    bugs = []
    for bug in stream_jsonl(str(input_file)):
        bugs.append(bug)

    # Perform manual analysis
    for bug in tqdm.tqdm(bugs):
        try:
            # Skip bugs without prompt
            if bug["evaluation"] == None:
                continue

            # Skip bugs that are already semantical matches
            if any(x["exact_match"] or x["ast_match"] for x in bug["evaluation"]):
                continue

            # Skip bugs that do not have plausible patches
            if not any(x["test"] for x in bug["evaluation"]):
                continue

            for evaluation in bug["evaluation"]:
                # Skip if the patch is not plausible
                if not evaluation["test"]:
                    continue

                patch = evaluation["generation"]
                comentless_patch = remove_java_comments(patch)
                cache_filename = Path(cache_path, hashlib.sha256(comentless_patch.encode() + bug["identifier"].encode()).hexdigest())

                result = 0
                
                # If we have already analyzed this patch, we load the result
                if os.path.exists(cache_filename):
                    print(f"Loading {cache_filename}")
                    with open(cache_filename) as f:
                        result = int(json.loads(f.read())["result"])
                # Otherwise we ask the user
                else:
                    # Print ground truth and plausible patch side-by-side in the terminal
                    ground_truth = compute_diff(
                        remove_empty_lines(remove_java_comments(bug["buggy_code"].strip())),
                        remove_empty_lines(
                            remove_java_comments(fixed_functions[bug["identifier"]].strip())
                        ),
                    )
                    plausible_patch = compute_diff(
                        remove_empty_lines(remove_java_comments(bug["buggy_code"].strip())),
                        remove_empty_lines(
                            remove_java_comments(evaluation["generation"].strip())
                        ),
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

                    # Write the result to the cache
                    with open(cache_filename,"w") as f:
                        f.write(json.dumps({"identifier":bug["identifier"],"patch":patch,"result":result}))

                    # Clear the terminal
                    os.system("cls" if os.name == "nt" else "clear")

                # Update the bug with the result
                if result == 1:
                    evaluation["semantical_match"] = True
                else:
                    evaluation["semantical_match"] = False
        finally:
            # Write the bug to the output file
            with open(output_file, "a+") as f:
                f.write(json.dumps(bug) + "\n")


def manual_analysis(user: str):
    if user == "andre":
        results_path = f"{script_dir}/../../results/1_andre"
        cache_path = f"{script_dir}/../../results/1_andre/cache"
    elif user == "sen":
        results_path = f"{script_dir}/../../results/1_sen"
        cache_path = f"{script_dir}/../../results/1_sen/cache"

    for input_file in Path(f"{script_dir}/../../results/0_original").glob("*.jsonl"):
        output_file = Path(results_path, input_file.name.replace(".jsonl", f"_{user}.jsonl"))
        manual_analysis_file(input_file, output_file, cache_path)

    for input_file in Path(f"{script_dir}/../../results/0_original").glob("*.jsonl.gz"):
        output_file = Path(results_path, input_file.name.replace(".jsonl.gz", f"_{user}.jsonl"))
        manual_analysis_file(input_file, output_file, cache_path)

    if os.path.exists(output_file):
        os.remove(output_file)

if __name__ == "__main__":
    fire.Fire(manual_analysis)
