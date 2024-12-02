import json
import tqdm
import fire
import copy
import os

from pathlib import Path

script_dir = os.path.dirname(__file__)

def merge_analysis_file(andre_file: str, sen_file: str, output_file: str):
    # Read all bugs from André's file
    andre_bugs = []
    with open(andre_file) as f:
        for line in f.readlines():
            bug = json.loads(line)
            andre_bugs.append(bug)

    # Read all bugs from Sen's file
    sen_bugs = []
    with open(sen_file) as f:
        for line in f.readlines():
            loaded_bug = json.loads(line)
            sen_bugs.append(loaded_bug)

    # Assert that both files have the same number of bugs
    assert len(andre_bugs) == len(
        sen_bugs
    ), f"The number of bugs in both files is different: {len(andre_bugs)} vs {len(sen_bugs)}"
    # Assert that the bugs are in the same order
    for i in range(len(andre_bugs)):
        assert (
            andre_bugs[i]["identifier"] == sen_bugs[i]["identifier"]
        ), f"The bugs are not in the same order at index {i}: {andre_bugs[i]['identifier']} vs {sen_bugs[i]['identifier']}"

    # Merge both files
    bugs = []
    for andre_bug, sen_bug in tqdm.tqdm(zip(andre_bugs, sen_bugs)):
        bug = copy.deepcopy(andre_bug)
        # Skip if the bug has any AST or exact matched patch
        if bug["evaluation"] is None or any(
            x["exact_match"] or x["ast_match"] for x in bug["evaluation"] if x is not None
        ):
            bugs.append(bug)
            continue

        # Merge results
        for i, evaluation in enumerate(bug["evaluation"]):
            # Check that test results are the same
            if evaluation is not None and "semantical_match" in evaluation:
                if (
                    andre_bug["evaluation"][i]["semantical_match"]
                    != sen_bug["evaluation"][i]["semantical_match"]
                ):
                    print(
                        f"Test results are different for bug {andre_bug['identifier']} patch {i}: {andre_bug['evaluation'][i]['semantical_match']} vs {sen_bug['evaluation'][i]['semantical_match']}"
                    )
                    bug["evaluation"][i]["semantical_match"] = "Disagree"

        bugs.append(bug)

    # Write the bugs to the output file
    for bug in bugs:
        with open(output_file, "a+") as f:
            f.write(json.dumps(bug) + "\n")


def merge_analysis():
    andre_results_path = f"{script_dir}/../../results/1_andre"
    sen_results_path = f"{script_dir}/../../results/1_sen"
    results_path = f"{script_dir}/../../results/2_merged"

    for input_file in Path(andre_results_path).glob("*.jsonl"):
        sen_input_file = Path(sen_results_path, input_file.name.replace("_andre.jsonl", "_sen.jsonl"))
        output_file = Path(results_path, input_file.name.replace("_andre.jsonl", f"_merged.jsonl"))
        os.remove(output_file) if os.path.exists(output_file) else None
        merge_analysis_file(input_file, sen_input_file, output_file)

if __name__ == "__main__":
    fire.Fire(merge_analysis)