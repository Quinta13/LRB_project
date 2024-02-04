import os
import shutil
from io import BytesIO
from typing import List, Optional
from zipfile import ZipFile

import requests


def extract_zip_from_url(url: str, target_dir: str):
    """
    Download a ZIP file from a given URL and extract its contents to a specified directory.

    :param url: The URL of the ZIP file to download.
    :param target_dir: The directory where the contents of the ZIP file will be extracted.
    """

    # Make a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:

        # Create a ZipFile object from the downloaded content
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(target_dir)
        print(f"Successfully extracted contents to {target_dir}")

    else:
        print(f"Failed to download ZIP file. Status code: {response.status_code}")

def move_content_one_level_up(base_path: str):
    """
    Move the contents of a subdirectory one level up, effectively merging the subdirectory with its parent.

    :param base_path: The base path containing the subdirectory to be flattened.
    """

    # List all directories inside the base path
    subdirectories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    # Check if there's exactly one subdirectory
    if len(subdirectories) == 1:
        subdirectory_name = subdirectories[0]
        subdirectory_path = os.path.join(base_path, subdirectory_name)

        # Move the content of the subdirectory to the base path
        for item in os.listdir(subdirectory_path):
            source_path = os.path.join(subdirectory_path, item)
            destination_path = os.path.join(base_path, item)
            shutil.move(source_path, destination_path)

        # Remove the now-empty subdirectory
        os.rmdir(subdirectory_path)
        print(f"Content moved from '{subdirectory_name}' to '{base_path}' and directory removed.")
    else:
        print("Error: Unexpected directory structure.")

def read_txt_file(file_path: str) -> Optional[List[str]]:
    """
    Read the contents of a text file and return a list of lines.

    :param file_path: The path to the text file.
    :return: A list of strings representing the lines in the text file.
             Returns None if the file is not found or an error occurs during reading.
    """

    try:
        # Read content
        with open(file_path, 'r') as file:
            lines = [line.rstrip('\n') for line in file.readlines()]
            return lines

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def write_txt_file(file_path: str, content: str):
    """
    Write the given content to a text file.

    :param file_path: The path to the text file.
    :param content: The content to be written to the file.
    """

    try:
        # Write content
        with open(file_path, 'w') as file:
            file.write(content)
        print(f"Content successfully written to {file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def words_to_txt_file(words: List[str], file_path: str):
    """
    Write a string of words to a text file, with each word on a new line.

    :param words: The string containing words to be written to the file.
    :param file_path: The path to the text file where the words will be written.
    """

    words_content = "".join([f"{word}\n" for word in words])
    write_txt_file(file_path=file_path, content=words_content)



def print_list(l: List, title: str):
    """
    Print the elements of a list with a title and the total count.

    :param l: The list to be printed.
    :param title: The title to be displayed before printing the list.
    """

    print(title)
    for el in l:
        print(f"- {el}")

    print(f"Tot: {len(l)}")


