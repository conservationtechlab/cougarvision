import os
import re


def get_last_file_number(folder_path):
    max_num = 0
    for filename in os.listdir(folder_path):
        # Extract digits from the filename using regex
        num = re.findall(r'\d+', filename)
        if num:  # If there are digits in the filename
            max_num = max(max_num, int(num[-1]))  # Use the last set of digits as the number
    return max_num

