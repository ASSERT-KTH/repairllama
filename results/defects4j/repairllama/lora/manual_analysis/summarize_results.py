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

sen_bugs = {}
andre_bugs = {}

def get_semantical_match(bugs):
    semantical = set()
    for bug_id in bugs:
        if "Semantical match" in bugs[bug_id]["test_results"]:
            semantical.add(bug_id)
    return semantical

with open("sen_d4j_check_results_total.json") as f:
    for line in f.readlines():
        bug = json.loads(line)
        sen_bugs[bug["bug_id"]] = bug

with open("andre_d4j_check_result_total.json") as f:
    for line in f.readlines():
        bug = json.loads(line)
        andre_bugs[bug["bug_id"]] = bug

semantical_sen = get_semantical_match(sen_bugs)
semantical_andre = get_semantical_match(andre_bugs)

print(f"Sen found {len(semantical_sen)} semantically fixed bugs")
print(f"André found {len(semantical_andre)} semantically fixed bugs")

print(f"Sen found {len(semantical_sen-semantical_andre)} unique bugs")
print(semantical_sen-semantical_andre)
print(f"André found {len(semantical_andre-semantical_sen)} unique bugs")
print(semantical_andre-semantical_sen)

diff = semantical_andre-semantical_sen
for bug_id in diff:
    bug = andre_bugs[bug_id]
    for i, patch in enumerate(bug["patches"]):
        if bug["test_results"][i] == "Semantical match":
            print("GROUND TRUTH:")
            print(compute_diff(bug["buggy_code"], bug["fixed_code"]))
            print("PLAUSIBLE PATCH:")
            print(compute_diff(bug["buggy_code"], patch))
            print(bug_id)
            input()
