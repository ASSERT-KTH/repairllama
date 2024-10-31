from typing import List, Tuple

import re

def fault_localizer(function: str) -> List[Tuple[int, int]]:
    """
    RepairLLaMA's fault localization algorithm.
    
    Given a function, the localizer returns all possible regions in the function.
    Each region is a tuple of two line numbers.
    start and end are both inclusive.

    Args:
        function: The function to be localized.
    Returns:
        A list of tuples, where each tuple contains two line numbers representing the buggy region.
    """
    # Split the function into lines
    lines = function.splitlines()
    total_lines = len(lines)
    regions = []
    
    # Generate all possible regions (line pairs)
    for start in range(1, total_lines):
        for end in range(start, total_lines):
            # Skip empty regions
            if start == end:
                continue
            
            # Add region as tuple of (start_line, end_line), both inclusive
            regions.append((start, end - 1))
    
    return regions

def print_fault_localization(function: str, region: Tuple[int, int]):
    lines = function.splitlines(keepends=True)
    pre_region = lines[:region[0]]
    buggy_region = lines[region[0]:region[1]+1]
    post_region = lines[region[1]+1:]

    result = ""
    for line in pre_region:
        result += line
    # Extract indentation using regex pattern
    indent_match = re.match(r'^(\s*)', buggy_region[0])
    indentation = indent_match.group(1) if indent_match else ""
    result += indentation + "// buggy code\n"
    for line in buggy_region:
        result += indentation + f"// {line.strip()}\n"
    result += indentation + "<FILL_ME>\n"
    for line in post_region:
        result += line
    return result
