import json

import argparse

from difflib import unified_diff
from generate_input_repr_data import pure_buggy_function_with_fault_location, pure_buggy_function_with_cloze_prompt, instruct_prompt
from utils import remove_java_comments_improved, remove_space_lines, extract_buggy_start_end_line


def main():

    # Parse arguments
    parser = argparse.ArgumentParser("Build dataset with different input-output representations for automated program repair")
    parser.add_argument("--meta_data_path", '-md', type=str, required=True, help="Path to the meta data.")
    parser.add_argument("--output_path", '-o', type=str, required=True, help="Path to the output.")
    parser.add_argument("--input_representation", '-ir', type=str, required=True, choices=['pbf', 'pbfwfl', 'pbfwc', 'pbfwln', 'pbfwcp', 'pbfwcpblc', 'instruct'],
                        help="""
                                Pure buggy function: pbf/IR1,
                                Pure buggy function with fault location information: pbfwfl/IR2,
                                Pure buggy function with comments: pbfwc,
                                Pure buggy function with line number: pbfwln,
                                Pure buggy function with cloze prompt: pbfwcp/IR3,
                                Pure buggy function with cloze prompt and buggy lines comments: pbfwcpblc/IR4,
                                Instruct prompt (GPT3.5/4): instruct,
                            """
    )

    args = parser.parse_args()
    with open(args.meta_data_path, 'r') as f:
        meta_bugs = [json.loads(line) for line in f.readlines()]

    std_data = []

    for sample in meta_bugs:

        if sample["buggy_code"] is None or sample["fixed_code"] is None:
            std_data.append(sample)
            continue

        # get buggy function and fixed function
        buggy_func = sample['buggy_code']
        fixed_func = sample['fixed_code']

        if args.input_representation != "pbdwc":
            # remove comments and space lines
            buggy_func_clean = remove_space_lines(remove_java_comments_improved(buggy_func))
            fixed_func_clean = remove_space_lines(remove_java_comments_improved(fixed_func))

            if args.input_representation in {"pbfwfl", "pbfwcp", "pbfwcpblc", "instruct"}:

                # get diff lines
                diff_lines = list(unified_diff(
                    buggy_func_clean.splitlines(keepends=True), 
                    fixed_func_clean.splitlines(keepends=True),
                    n=1000000,
                    )
                )

                # get buggy start and end line
                bug_start_line, bug_end_line = extract_buggy_start_end_line(diff_lines)
        
        if args.input_representation == "pbf":

            prompt = buggy_func_clean

        elif args.input_representation == "pbfwfl":

            # get buggy function with fault location
            prompt = pure_buggy_function_with_fault_location(diff_lines, bug_start_line, bug_end_line)

        elif args.input_representation == "pbfwc":

            prompt = buggy_func

        elif args.input_representation == "pbfwln":

            buggy_func_lines = buggy_func_clean.splitlines(keepends=True)
            buggy_func_lines_with_line_number = [str(idx + 1) + " " + line for idx, line in enumerate(buggy_func_lines)]
            prompt = ''.join(buggy_func_lines_with_line_number)

        elif args.input_representation == "pbfwcp":

            # get buggy function with fault location
            prompt = pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=False)

        elif args.input_representation == "pbfwcpblc":

            # get buggy function with fault location
            prompt = pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=True)

        elif args.input_representation == "instruct":

            prompt = instruct_prompt(diff_lines, bug_start_line, bug_end_line)

        else:
            raise ValueError("Invalid input representation.")

        sample['prompt'] = prompt

        std_data.append(sample)
    
    # Save the data
    with open(args.output_path, 'w') as f:
        for sample in std_data:
            f.write(json.dumps(sample) + '\n')
            

if __name__ == "__main__":
    main()

