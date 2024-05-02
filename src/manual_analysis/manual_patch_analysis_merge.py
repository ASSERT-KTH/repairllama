import json
import tqdm
import fire
import copy


def merge_analysis(andre_file: str, sen_file: str, output_file: str):
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

    # Sort bugs by bug_id
    andre_bugs = sorted(andre_bugs, key=lambda x: x["bug_id"])
    sen_bugs = sorted(sen_bugs, key=lambda x: x["bug_id"])

    # Assert that both files have the same number of bugs
    assert len(andre_bugs) == len(
        sen_bugs
    ), f"The number of bugs in both files is different: {len(andre_bugs)} vs {len(sen_bugs)}"
    # Assert that the bugs are in the same order
    for i in range(len(andre_bugs)):
        assert (
            andre_bugs[i]["bug_id"] == sen_bugs[i]["bug_id"]
        ), f"The bugs are not in the same order at index {i}: {andre_bugs[i]['bug_id']} vs {sen_bugs[i]['bug_id']}"

    # Merge both files
    bugs = []
    for andre_bug, sen_bug in tqdm.tqdm(zip(andre_bugs, sen_bugs)):
        bug = copy.deepcopy(andre_bug)
        # Skip if the bug has any AST or exact matched patch
        if "AST match" in bug["test_results"] or "Line match" in bug["test_results"]:
            bugs.append(bug)
            continue

        # Merge results
        for i, _ in enumerate(bug["patches"]):
            # Check that test results are the same
            if andre_bug["test_results"][i] != sen_bug["test_results"][i]:
                print(
                    f"Test results are different for bug {andre_bug['bug_id']} patch {i}: {andre_bug['test_results'][i]} vs {sen_bug['test_results'][i]}"
                )
                bug["test_results"][i] = "Disagree"

        bugs.append(bug)

    # Write the bugs to the output file
    for bug in bugs:
        with open(output_file, "a+") as f:
            f.write(json.dumps(bug) + "\n")


if __name__ == "__main__":
    fire.Fire(merge_analysis)
