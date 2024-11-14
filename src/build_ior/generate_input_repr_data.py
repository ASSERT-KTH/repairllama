import re


def pure_buggy_function_with_fault_location(diff_lines, bug_start_line, bug_end_line):
    """
    In this approach, we regard all single-function as a single-chunk bug. We add three special lines to the buggy function:
    1. "// buggy lines start:\n": This line is used to indicate the start of buggy lines.
    2. "// buggy lines end\n": This line is used to indicate the end of buggy lines.
    3. "// fixed lines:\n": This line is used to indicate the start of fixed lines.
    """
    input_data = []  
    buggy_chunk = []

    for code_line in diff_lines[bug_start_line:bug_end_line+1]:
        if code_line.startswith("-"):
            buggy_chunk.append(code_line[1:])
        elif code_line.startswith("+"):
            continue
        else:
            buggy_chunk.append(code_line[1:])
    
    # This is a special case, where the bug is a sngle-chunk and fixed by only adding continuous lines.
    if len(buggy_chunk) == 0:
        buggy_chunk = ["\n"]

    # For each diff line, we should start from the second character.
    diff_lines = [line[1:] for line in diff_lines]
    input_data = diff_lines[3:bug_start_line] + ["// buggy lines start:\n"] + buggy_chunk + ["// buggy lines end\n"] + diff_lines[bug_end_line+1:] + ["// fixed lines:\n"]

    return "".join(input_data)


def pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=True):
    """
    In this approach, we regard all single-function as a single-chunk bug. We replace the buggy lines with a cloze prompt (mask token).
    """

    def get_beginning_spaces(s):
        match = rhe.match(r'^[\s\t]+', s)
        return match.group(0) if match else None

    mask_token = "<FILL_ME>\n"
    input_data = []
    
    # Get the beginning spaces of the first buggy line.
    beginning_spaces = get_beginning_spaces(diff_lines[bug_start_line][1:])
    mask_token = beginning_spaces + mask_token if beginning_spaces else mask_token

    if include_buggy_lines:

        # Get the buggy lines.
        diff_chunk = diff_lines[bug_start_line:bug_end_line+1]
        buggy_chunk = []
        for code_line in diff_chunk:
            if code_line.startswith("-"):
                buggy_chunk.append(code_line[1:])
            elif code_line.startswith("+"):
                continue
            else:
                buggy_chunk.append(code_line[1:])
        
        # This is a special case, where the bug is a sngle-chunk and fixed by only adding continuous lines.
        if len(buggy_chunk) == 0:
            buggy_chunk = ["\n"]
        
        # Comment each buggy line.
        comment_buggy_chunk =["// buggy code\n"] + ["// " + line for line in buggy_chunk]

        diff_lines = [line[1:] for line in diff_lines]
        input_data = diff_lines[3:bug_start_line] + comment_buggy_chunk + [mask_token] + diff_lines[bug_end_line+1:]       
    else:
        diff_lines = [line[1:] for line in diff_lines]
        input_data = diff_lines[3:bug_start_line] + [mask_token] + diff_lines[bug_end_line+1:]

    return "".join(input_data)

def instruct_prompt(diff_lines, bug_start_line, bug_end_line):
    return f"""There’s a bug in the Java program below. Try to fix it and return the complete fix for the code in the form of the markdown code block. Generate the code to replace the <FILL_ME> token.

```java
{pure_buggy_function_with_cloze_prompt(diff_lines, bug_start_line, bug_end_line, include_buggy_lines=True)}
```
"""