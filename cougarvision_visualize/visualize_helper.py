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
    
    
def create_folder(folder_path):
    """
    Check if a folder exists at the specified path, and create it if it doesn't.

    folder_path: The path of the folder to check and potentially create.
    return: path
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

