import json
import random

import pandas as pd
import argparse

from pathlib import Path
from difflib import unified_diff
from transformers import AutoTokenizer # We may can deal with length in fine-tuning step
from concurrent.futures import ThreadPoolExecutor, as_completed
from generate_input_repr_data import pure_buggy_function_with_fault_location, pure_buggy_function_with_cloze_prompt
from generate_output_repr_data import fixed_lines_output, fixed_diff_output, fixed_diff_output_with_line_number
from tqdm import tqdm
from utils import remove_java_comments_improved, remove_space_lines, extract_buggy_start_end_line


def generate_input_repr_data(meta_data, input_representation):
    
    inputs_data = {}
    
    for idx, sample in tqdm(enumerate(meta_data.iloc)):

        # Extract buggy function and fixed function
        buggy_function = sample['buggy_function']
        fixed_function = sample['fixed_function']

        # Remove comments and space lines if input representation is not pbfwc
        if input_representation != "pbfwc" and input_representation != "pbfwcpblcc":
            norm_buggy_function = remove_space_lines(remove_java_comments_improved(buggy_function))
            norm_fixed_function = remove_space_lines(remove_java_comments_improved(fixed_function))

            if norm_buggy_function == norm_fixed_function:
                inputs_data[sample['diff']] = 'Bug fixing by modifying comments.'
                continue
        
        if input_representation == "pbfwfl" or input_representation == "pbfwcpblcc":
            buggy_lines = buggy_function.splitlines(keepends=True)
            fixed_lines = fixed_function.splitlines(keepends=True)
        else:
            buggy_lines = norm_buggy_function.splitlines(keepends=True)
            fixed_lines = norm_fixed_function.splitlines(keepends=True)

        # Generate diff
        diff = "".join(unified_diff(buggy_lines, fixed_lines, n=1000000))
        diff_lines = diff.splitlines(True)
        # Get the diff location
        bug_start_line, bug_end_line = extract_buggy_start_end_line(diff_lines)

        if input_representation == "pbfwfl":  # This input representation is only suitable for single-chunk bugs

            input_data = pure_buggy_function_with_fault_location(diff_lines, bug_start_line, bug_end_line)

        elif input_representation == "pbf":

            input_data = norm_buggy_function

        elif input_representation == "pbfwln":

            input_data = "".join([str(idx+1) + " " + line for idx, line in enumerate(buggy_lines)])

        elif input_representation == "pbfwc":

            input_data = buggy_function

        elif input_representation == "pbfwcp":

            input_data = pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=False)

        elif input_representation == "pbfwcpblc":

            input_data = pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=True)

        elif input_representation == "pbfwcpblcc":

            input_data = pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=True)

        else:
            raise ValueError("Invalid input representation!")
        # print(diff)
        # print(input_data)
        # if idx == 4:
        #     breakpoint()
        inputs_data[sample['diff']] = input_data

    return inputs_data


def generate_output_repr_data(meta_data, input_representation, output_representation):

    outputs_data = {}

    for idx, sample in tqdm(enumerate(meta_data.iloc)):

        # Extract buggy function and fixed function
        buggy_function = sample['buggy_function']
        fixed_function = sample['fixed_function']

        # Remove comments and space lines if input representation is not pbfwc
        if input_representation != "pbfwc" and input_representation != "pbfwcpblcc":
            norm_buggy_function = remove_space_lines(remove_java_comments_improved(buggy_function))
            norm_fixed_function = remove_space_lines(remove_java_comments_improved(fixed_function))

            if norm_buggy_function == norm_fixed_function:
                outputs_data[sample['diff']] = 'Bug fixing by modifying comments.'
                continue
        
        if input_representation == "pbfwfl" or input_representation == "pbfwcpblcc":
            buggy_lines = buggy_function.splitlines(keepends=True)
            fixed_lines = fixed_function.splitlines(keepends=True)
        else:
            buggy_lines = norm_buggy_function.splitlines(keepends=True)
            fixed_lines = norm_fixed_function.splitlines(keepends=True)

        # if idx == 216:
        #     print(buggy_function)
        #     breakpoint()

        if output_representation == "fl":
            
            # Generate diff
            diff = "".join(unified_diff(buggy_lines, fixed_lines, n=1000000))
            diff_lines = diff.splitlines(True)

            bug_start_line, bug_end_line = extract_buggy_start_end_line(diff_lines)
            output_data = fixed_lines_output(diff_lines, bug_start_line, bug_end_line)

        elif output_representation == "ff":
            
            if input_representation != "pbfwln":
                output_data = norm_fixed_function if input_representation != "pbfwc" else fixed_function
            else:
                output_data = "".join([str(idx+1) + " " + line for idx, line in enumerate(fixed_lines)])

        elif output_representation == "ldw/oc":

            if input_representation != "pbfwln":
                output_data, _ = fixed_diff_output(buggy_lines, fixed_lines, context_lines=0)
            else:
                output_data, _ = fixed_diff_output_with_line_number(buggy_lines, fixed_lines, context_lines=0)

        elif output_representation == "ldwolc":

            if input_representation != "pbfwln":
                output_data, _ = fixed_diff_output(buggy_lines, fixed_lines, context_lines=1)
            else:
                output_data, _ = fixed_diff_output_with_line_number(buggy_lines, fixed_lines, context_lines=1)

        elif output_representation == "ldwtlc":

            if input_representation != "pbfwln":
                output_data, _ = fixed_diff_output(buggy_lines, fixed_lines, context_lines=3)
            else:
                output_data, _ = fixed_diff_output_with_line_number(buggy_lines, fixed_lines, context_lines=3)

        else:
            raise ValueError("Invalid output representation!")

        # print(_)
        # print("------------------")
        # print(output_data)
        # print("-------New one--------")
        # if idx == 9:
        #     breakpoint()
        
        outputs_data[sample['diff']] = output_data

    return outputs_data


def build_dataset_for_ior(meta_data, input_representation, output_representation, output_path):
    
    # Extract buggy function and fixed function and label
    # data_list = [(sample['diff'], sample['buggy_function'], sample['fixed_function']) for sample in meta_data.iloc]
    # print(len(data_list))
    
    input_data = generate_input_repr_data(meta_data, input_representation)
    output_data = generate_output_repr_data(meta_data, input_representation, output_representation)

    # build meta data for input-output representation
    meta_data = {}
    for diff in input_data.keys():
        meta_data[diff] = {
            'input': input_data[diff],
            'output': output_data[diff]
        }
    
    # build training and testing data
    full_data = []
    for diff in meta_data.keys():
        if input_data[diff] == "Bug fixing by modifying comments.":
            continue
        full_data.append(
            {
                'input': input_data[diff],
                'output': output_data[diff],
            }
        )
    
    # shuffle data
    random.shuffle(full_data)
    # split 1000 samples for testing
    train_data = full_data[1000:]
    test_data = full_data[:1000]

    # save data
    with open(output_path + "/meta_data.json", 'w') as f:
        json.dump(meta_data, f, indent=4)
    with open(output_path + "/train_data.jsonl", 'w') as f:
        for each_data in train_data:
            f.write(json.dumps(each_data) + "\n")
    with open(output_path + "/test_data.jsonl", 'w') as f:
        for each_data in test_data:
            f.write(json.dumps(each_data) + "\n")
    with open(output_path + "/README.md", 'a') as f:
        f.write("\n" + "The size of training data is: " + str(len(train_data)) + "\n")
        f.write("The size of testing data is: " + str(len(test_data)) + "\n")
    print("The size of training data is: ", len(train_data))
    print("The size of testing data is: ", len(test_data))
    print("Done!")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser("Build dataset with different input-output representations for automated program repair")
    parser.add_argument("--meta_data_path", '-md', type=str, required=True, help="Path to the meta data.")
    parser.add_argument("--output_path", '-o', type=str, required=True, help="Path to the output.")
    parser.add_argument("--input_representation", '-ir', type=str, required=True, choices=['pbf', 'pbfwfl', 'pbfwc', 'pbfwln', 'pbfwcp', 'pbfwcpblc', 'pbfwcpblcc'],
                        help="""
                                Pure buggy function: pbf,
                                Pure buggy function with fault location information: pbfwfl,
                                Pure buggy function with comments: pbfwc,
                                Pure buggy function with line number: pbfwln,
                                Pure buggy function with cloze prompt: pbfwcp,
                                Pure buggy function with cloze prompt and buggy lines comments: pbfwcpblc,
                                Pure buggy function with cloze prompt and buggy lines comments and comments: pbfwcpblcc,
                            """
    )
    parser.add_argument("--output_representation", '-or', type=str, required=True, choices=['fl', 'ldw/oc', 'ldwolc', 'ldwtlc', 'ff'],
                        help="""
                                Fixed lines: fl,
                                Line diff without context: ldw/oc,
                                Line diff with one line context: ldwolc,
                                Line diff with three line context: ldwtlc,
                                Full function: ff,
                            """
    )
    
    args = parser.parse_args()

    meta_data_path = Path(args.meta_data_path)

    # Find all meta data file
    meta_data_list = []
    for meta_data_file in meta_data_path.rglob("*.parquet"):
        meta_data_list.append(meta_data_file)
    
    # Use pandas to read all meta data
    meta_data = pd.concat([pd.read_parquet(meta_data_file) for meta_data_file in meta_data_list])
    # Remove sample whose fixed function is empty
    meta_data = meta_data[meta_data['fixed_function'].str.len() > 0]
    # Remove sample whose buggy function is the same as fixed function
    meta_data = meta_data[meta_data['buggy_function'] != meta_data['fixed_function']]
    # Delete duplicate samples.
    meta_data = meta_data.drop_duplicates(subset=['buggy_function', 'fixed_function'])

    # print head of meta data
    print(meta_data.shape)
    build_dataset_for_ior(meta_data, args.input_representation, args.output_representation, args.output_path)


if __name__ == "__main__":
    main()

