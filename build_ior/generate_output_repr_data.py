from difflib import unified_diff


def fixed_lines_output(diff_lines, bug_start_line, bug_end_line):
    """
    We extract the code lines and diff line starting with "+" as the fixed lines.
    """
    output_data = []

    for code_line in diff_lines[bug_start_line:bug_end_line+1]:
        if code_line.startswith("+"):
            output_data.append(code_line[1:])
        elif code_line.startswith("-"):
            continue
        else:
            output_data.append(code_line[1:])
    
    # This is a special case, where the bug is a sngle-chunk and fixed by only deleting continuous lines, the output data is empty.

    return "".join(output_data)


def fixed_func_output(diff_lines):
    """
    We extract the code lines and diff line starting with "+" as the fixed lines.
    """

    output_data = []

    for code_line in diff_lines:
        if code_line.startswith(("-", "---", "@@", "+++")):
            continue
        elif code_line.startswith("+"):
            output_data.append(code_line[1:])
        else:
            output_data.append(code_line[1:])
    
    # This is a special case, where the bug is a sngle-chunk and fixed by only deleting continuous lines, the output data is empty.

    return "".join(output_data)


def fixed_diff_output(buggy_lines, fixed_lines, context_lines):
    """
    We extract the diff of the buggy function and fixed function, with different number of context lines, as the output data.
    """

    output_data = []
    diff_index = [] # (a, b) a is the start line of the diff, b is the end line of the diff

    diff = "".join(unified_diff(buggy_lines, fixed_lines, n=1000000))
    diff_lines = diff.splitlines(True)[3:]

    i = 0
    start_flag = 0
    while i < len(diff_lines):
        if diff_lines[i].startswith(('+', '-')):
            start_flag = i
            while i < len(diff_lines):
                if diff_lines[i].startswith(('+', '-')):
                    i += 1
                else:
                    diff_index.append((start_flag, i))
                    i += 1
                    break
        else:
            i += 1

    for each_index in diff_index:

        diff_start, diff_end = each_index

        diff_hunk = diff_lines[diff_start:diff_end]

        # Get the pre_context_lines
        pre_context_lines = []
        # Get the maximum continuous non-diff lines before the diff
        for i in range(diff_start-1, diff_start-context_lines-1, -1):
            if i < 0:
                break
            if diff_lines[i].startswith(('+', '-')):
                break
            else:
                pre_context_lines.append(diff_lines[i])
        pre_context_lines.reverse()

        # Get the post_context_lines
        post_context_lines = []
        # Get the maximum continuous non-diff lines after the diff
        for i in range(diff_end, diff_end+context_lines, 1):
            if i > len(diff_lines) - 1:
                break
            if diff_lines[i].startswith(('+', '-')):
                break
            else:
                post_context_lines.append(diff_lines[i])

        context_diff = pre_context_lines + diff_hunk + post_context_lines
        output_data.append("".join(context_diff))
    
    if len(output_data) == 1:
        output_diff = output_data[0]
    else:
        output_diff = ""
        for i in range(len(output_data)):
            output_diff += output_data[i]
            if i < len(output_data) - 1:
                output_diff += "\n"
    
    return output_diff, "".join(diff_lines)


def fixed_diff_output_with_line_number(buggy_lines, fixed_lines, context_lines):
    """
    We extract the diff of the buggy function and fixed function, with different number of context lines, as the output data.
    We need to add the line number to the output diff on the basis of the function "fixed_diff_output"
    """

    output_data = []
    diff_index = [] # (a, b) a is the start line of the diff, b is the end line of the diff

    diff = "".join(unified_diff(buggy_lines, fixed_lines, n=1000000))
    diff_lines = diff.splitlines(True)[3:]

    i = 0
    start_flag = 0 # To point out the beginning of a diff chunk in diff_lines
    in_buggy_line_flag = 0
    buggy_line_start_num = 0 # To point out the line number of the first buggy line pre diff
    while i < len(diff_lines):
        if diff_lines[i].startswith(('+', '-')):
            buggy_line_start_num = in_buggy_line_flag
            if diff_lines[i].startswith('-'):
                in_buggy_line_flag += 1
            start_flag = i
            i += 1
            while i < len(diff_lines):
                if diff_lines[i].startswith(('+', '-')):
                    if diff_lines[i].startswith('-'):
                        in_buggy_line_flag += 1
                    i += 1
                else:
                    diff_index.append((start_flag, i, buggy_line_start_num))
                    in_buggy_line_flag += 1
                    i += 1
                    break
        else:
            in_buggy_line_flag += 1
            i += 1

    for each_index in diff_index:

        diff_start, diff_end, in_buggy_line_flag = each_index

        diff_hunk = diff_lines[diff_start:diff_end]

        # Get the pre_context_lines
        pre_context_lines = []
        # Get the maximum continuous non-diff lines before the diff
        for i in range(diff_start-1, diff_start-context_lines-1, -1):
            if i < 0:
                break
            if diff_lines[i].startswith(('+', '-')):
                break
            else:
                pre_context_lines.append(diff_lines[i])
        pre_context_lines.reverse()

        # Get the post_context_lines
        post_context_lines = []
        # Get the maximum continuous non-diff lines after the diff
        for i in range(diff_end, diff_end+context_lines, 1):
            if i > len(diff_lines) - 1:
                break
            if diff_lines[i].startswith(('+', '-')):
                break
            else:
                post_context_lines.append(diff_lines[i])

        # Add the line number to the pre_context_lines
        pre_context_lines_with_line_number = []
        # print(in_buggy_line_flag)
        for idx, line in enumerate(pre_context_lines):
            line = line[1:]
            index = buggy_lines.index(line)
            pre_context_lines_with_line_number.append(' ' + str(in_buggy_line_flag-len(pre_context_lines)+1+idx) + ' ' + line)

        # Add the line number to the diff_hunk
        diff_hunk_with_line_number = []
        i = 0
        delete_index = in_buggy_line_flag + 1
        add_index = in_buggy_line_flag + 1
        del_line_cnt = 0
        while i < len(diff_hunk):
            # Since this is a diff hunk, it must start with "+" or "-"
            if diff_hunk[i].startswith('-'):
                # it means this diff contains deleted lines
                line = diff_hunk[i][1:]
                diff_hunk_with_line_number.append('-' + str(delete_index) + ' ' + line)
                delete_index += 1
                i += 1
                del_line_cnt += 1
                while i < len(diff_hunk):
                    if diff_hunk[i].startswith('-'):
                        line = diff_hunk[i][1:]
                        diff_hunk_with_line_number.append('-' + str(delete_index) + ' ' + line)
                        delete_index += 1
                        i += 1
                        del_line_cnt += 1
                    else:
                        break
            else:
                # it means this diff contains added lines
                line = diff_hunk[i][1:]
                diff_hunk_with_line_number.append('+' + str(add_index) + ' ' + line)
                add_index += 1
                i += 1
                while i < len(diff_hunk):
                    if diff_hunk[i].startswith('+'):
                        line = diff_hunk[i][1:]
                        diff_hunk_with_line_number.append('+' + str(add_index) + ' ' + line)
                        add_index += 1
                        i += 1

        # Add the line number to the post_context_lines
        in_buggy_line_flag += del_line_cnt
        post_context_lines_with_line_number = []
        for idx, line in enumerate(post_context_lines):
            line = line[1:]
            index = buggy_lines.index(line)
            post_context_lines_with_line_number.append(' ' + str(in_buggy_line_flag+idx+1) + ' ' + line)

        context_diff = pre_context_lines_with_line_number + diff_hunk_with_line_number + post_context_lines_with_line_number
        output_data.append("".join(context_diff))
    
    if len(output_data) == 1:
        output_diff = output_data[0]
    else:
        output_diff = ""
        for i in range(len(output_data)):
            output_diff += output_data[i]
            if i < len(output_data) - 1:
                output_diff += "\n"
    
    diff_lines = [line[0] + str(idx+1) + line[1:] for idx, line in enumerate(diff_lines)]

    return output_diff, "".join(diff_lines)