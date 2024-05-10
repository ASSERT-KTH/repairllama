import json
import difflib
import tqdm
import fire
import gzip
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
    for bug in stream_jsonl(input_file):
        bugs.append(bug)

    # Perform manual analysis
    for bug in tqdm.tqdm(bugs, "Number of bugs processed: "):
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
                
                # Skip if semantical match is not disagree
                if evaluation["semantical_match"] != "Disagree":
                    continue

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
                result = int(input("Option: "))
                if result == 1:
                    evaluation["semantical_match"] = True
                    break

                # Clear the terminal
                os.system("cls" if os.name == "nt" else "clear")
        finally:
            # Write the bug to the output file
            with open(output_file, "a+") as f:
                f.write(json.dumps(bug) + "\n")


if __name__ == "__main__":
    fire.Fire(manual_analysis)
