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

def show_diff(input_file, bug_id):
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

    bugs = {}
    with open(input_file) as f:
        for line in f.readlines():
            bug = json.loads(line)
            bugs[bug["bug_id"]] = (bug)

    ground_truth = compute_diff(
        remove_empty_lines(remove_java_comments(bugs[bug_id]["buggy_code"].strip())),
        remove_empty_lines(
            remove_java_comments(fixed_functions[bug_id].strip())
        ),
    )

    print("Ground truth:")
    print(
        highlight(
            ground_truth, get_lexer_by_name("diff"), TerminalFormatter()
        )
    )

if __name__ == "__main__":
    fire.Fire(show_diff)