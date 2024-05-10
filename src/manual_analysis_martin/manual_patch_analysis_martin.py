import json
import difflib
import tqdm
import fire
import os
import re

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
    return "".join(
        list(
            difflib.unified_diff(
                buggy_code.splitlines(keepends=True),
                fixed_code.splitlines(keepends=True),
                n=context_len,
            )
        )
    )


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
    for bug in tqdm.tqdm(bugs, "Number of bugs processed: "):
        try:
            # Skip if the bug does not have any plausible patch
            if "Disagree" not in bug["test_results"]:
                continue
            # Skip if the bug has any AST or exact matched patch
            if (
                "AST match" in bug["test_results"]
                or "Line match" in bug["test_results"]
                or "AST Match" in bug["test_results"]
                or "Line Match" in bug["test_results"]
                or "Semantical match" in bug["test_results"]
            ):
                continue

            for i, patch in enumerate(bug["patches"]):
                # Skip if the patch is not plausible
                if bug["test_results"][i] != "Disagree":
                    continue

                # Print ground truth and plausible patch side-by-side in the terminal
                ground_truth = compute_diff(
                    remove_empty_lines(remove_java_comments(bug["buggy_code"].strip())),
                    remove_empty_lines(
                        remove_java_comments(fixed_functions[bug["bug_id"]].strip())
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
                result = int(input("Option: "))
                # If we find a semantic match we break the loop
                if result == 1:
                    bug["test_results"][i] = "Semantical match"
                    break

                # Clear the terminal
                os.system("cls" if os.name == "nt" else "clear")
        finally:
            # Write the bug to the output file
            with open(output_file, "a+") as f:
                f.write(json.dumps(bug) + "\n")


if __name__ == "__main__":
    fire.Fire(manual_analysis)
