import json
import fire

def merge_jsonl(input_file, output_file):
    bugs = {}
    with open(input_file) as f:
        for i, line in enumerate(f.readlines()):
            bug = json.loads(line)
            if bug["identifier"] in bugs:
                # Check if the stored bug is the same
                if bugs[bug["identifier"]] != bug:
                    print(f"Bug {bug['identifier']} is not the same in line {i}")
                    print(f"Stored bug: {bugs[bug['identifier']]}")
                    print(f"Current bug: {bug}")
                else:
                    continue
            else:
                bugs[bug["identifier"]] = bug
    
    with open(output_file, "w") as f:
        for bug in bugs.values():
            f.write(json.dumps(bug))
            f.write("\n")

if __name__ == '__main__':
    fire.Fire(merge_jsonl)