import json
import difflib
import fire
import os

from typing import Optional, List
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter

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
    return "".join(list(
        difflib.unified_diff(
            buggy_code.splitlines(keepends=True),
            fixed_code.splitlines(keepends=True),
            n=context_len,
        )
    ))


def manual_analysis(input_file: str, output_file: str):
    """
    Perform manual analysis on the bugs in the input file and write the results to the output file.

    Args:
    - input_file: str: The path to the input file containing the bugs.
    - output_file: str: The path to the output file to write the results.
    """
    # Load fixed functions
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

    # Read all bugs from the input_file
    bugs = []
    with open(input_file) as f:
        for line in f.readlines():
            bug = json.loads(line)
            bugs.append(bug)

    # Perform manual analysis
    for bug in bugs:
        try:
            # Skip if the bug does not have any plausible patch
            if "Plausible" not in bug["test_results"]:
                continue
            # Skip if the bug has any AST or exact matched patch
            if "AST match" in bug["test_results"] or "Line match" in bug["test_results"]:
                continue

            for i, patch in enumerate(bug["patches"]):
                # Skip if the patch is not plausible
                if bug["test_results"][i] != "Plausible":
                    continue 

                # Print ground truth and plausible patch side-by-side in the terminal
                ground_truth = compute_diff(bug["buggy_code"].strip(), fixed_functions[bug["bug_id"]].strip())
                plausible_patch = compute_diff(bug["buggy_code"].strip(), patch.strip())

                print("Ground truth:")
                print(highlight(ground_truth, get_lexer_by_name("diff"), TerminalFormatter()))
                print("Plausible patch:")
                print(highlight(plausible_patch, get_lexer_by_name("diff"), TerminalFormatter()))

                # Ask the user for the result
                print("0 -> different, 1 -> equivalent")
                result = int(input("Option: "))
                if result == 1:
                    bug["test_results"][i] = "Semantical match"

                # Clear the terminal
                os.system('cls' if os.name == 'nt' else 'clear')
        finally:
            # Write the bug to the output file
            with open(output_file, "a+") as f:
                f.write(json.dumps(bug) + "\n")

if __name__ == "__main__":
    fire.Fire(manual_analysis)