import re


def remove_java_comments(source: str) -> str:
    """Remove all single-line and multi-line comments from Java source code."""
    # This regex matches:
    # 1. Multi-line comments: /* ... */
    # 2. Single line comments: //...
    # 3. String literals: "..."
    # 4. Character literals: '...'
    pattern = r'/\*.*?\*/|//.*?(\n|$)|("([^"\\]*(?:\\.[^"\\]*)*)")|(\'([^\'\\]*(?:\\.[^\'\\]*)*)\')'
    
    def replacer(match):
        # If the match is a string literal or character literal, we return it as is
        if match.group(2) or match.group(4):
            return match.group(0)
        # If the match is a comment (single or multi-line), we return an empty string to remove it
        return ""
    
    return re.sub(pattern, replacer, source, flags=re.DOTALL)


def remove_java_comments_improved(source: str) -> str:
    # Define states
    NORMAL, SINGLE_COMMENT, MULTI_COMMENT, STRING_LITERAL, CHAR_LITERAL = range(5)
    
    state = NORMAL
    result = []
    i = 0
    
    while i < len(source):
        # Check the current state and process accordingly
        if state == NORMAL:
            if source[i:i+2] == "//":
                state = SINGLE_COMMENT
                i += 2
            elif source[i:i+2] == "/*":
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
            if source[i:i+2] == "*/":
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
                
    return ''.join(result)


def remove_space_lines(source):
    """Remove all space lines from Java source code."""
    return re.sub(r"^\s*$\n", "", source, flags=re.MULTILINE)


def extract_buggy_start_end_line(diff_lines):

    bug_start_line = 1000000
    bug_end_line = -1000000

    for idx, code_line in enumerate(diff_lines):
        if code_line.startswith(("---", "@@", "+++")):
            continue
        elif code_line.startswith(("+", "-")):
            if bug_start_line > idx:
                bug_start_line = idx
            if bug_end_line < idx:
                bug_end_line = idx
        else:
            continue
    return bug_start_line, bug_end_line

