import fire
import json
import tqdm
import hashlib
import os

from pathlib import Path

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

def extract_cache_from_file(file_path, cache_path):
    print(file_path)
    with file_path.open("r") as f:
        bugs = [json.loads(i) for i in f.readlines()]

    conflicted = set()

    for bug in tqdm.tqdm(bugs):
        # Skip if the bug has any AST or exact matched patch
        if bug["evaluation"] is None or any(
            x["exact_match"] or x["ast_match"] for x in bug["evaluation"]
        ):
            continue

        # Iterate over patches
        for evaluation in bug["evaluation"]:
            if evaluation["test"]:
                patch = evaluation["generation"]
                comentless_patch = remove_java_comments(patch)
                result = evaluation["semantical_match"]
                cache_filename = Path(cache_path, hashlib.sha256(comentless_patch.encode() + bug["identifier"].encode()).hexdigest())
                if cache_filename in conflicted:
                    print(f"Skipping {cache_filename} because it is conflicted")
                    continue
                elif cache_filename.exists():
                    # check that the stored result is the same
                    with open(cache_filename,"r") as f:
                        cache = json.loads(f.read())
                        if cache["result"] != result:
                            conflicted.add(cache_filename)
                            os.remove(cache_filename)
                            print(f"Error: {cache_filename} has different result: {cache['result']} vs {result}")
                else:
                    with open(cache_filename,"w") as f:
                        f.write(json.dumps({"identifier":bug["identifier"],"patch":patch,"result":result}))


def extract_cache(user: str):
    if user == "andre":
        results_path = "results/1_andre"
        cache_path = "results/1_andre/cache"
    elif user == "sen":
        results_path = "results/1_sen"
        cache_path = "results/1_sen/cache"
    else:
        raise ValueError(f"Unknown user: {user}")

    Path(cache_path).mkdir(exist_ok=True, parents=True)
    for i in Path(results_path).glob("*.jsonl"):
        extract_cache_from_file(i, cache_path)


if __name__ == "__main__":
    fire.Fire(extract_cache)
