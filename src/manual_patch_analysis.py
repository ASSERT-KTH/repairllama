import json
import difflib
import os

from typing import Optional, List

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

with open("sen_human_check.json") as f:
    for line in f.readlines():
        bug = json.loads(line)
        manual_results = []
        for i, patch in enumerate(bug["patches"]):
            print("GROUND TRUTH:")
            print(compute_diff(bug["buggy_code"], bug["fixed_code"]))
            print("PLAUSIBLE PATCH:")
            print(compute_diff(bug["buggy_code"], patch))
            print("RESULT (0 -> different, 1 -> doubt, 2 -> equivalent)")
            result = int(input())
            if result == 2:
                bug["test_results"][i] = "Semantical match"
                break
        os.system('cls' if os.name == 'nt' else 'clear')
        with open("andre_human_check_result_total.json", "a+") as f:
            f.write(json.dumps(bug) + "\n")
