import os
import shutil
from io import BytesIO
from typing import List
from zipfile import ZipFile
from numpy import mean
import requests
import matplotlib.pyplot as plt
from model import Result
from typing import Dict


def extract_zip_from_url(url: str, target_dir: str):

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

def read_txt_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = [line.rstrip('\n') for line in file.readlines()]
            return lines

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

class DRCDownloader:

    def __init__(self, path_: str, url: str):

        self._path: str = path_
        self._url: str = url

    def __str__(self) -> str:
        return f"DRCDownloader[path: {self.path}; url: {self.url}; {'' if self.is_downloaded else 'not'} downloaded]"

    def __repr__(self) -> str:
        return str(self)

    @property
    def path(self) -> str:
        return self._path

    @property
    def url(self) -> str:
        return self._url

    @property
    def is_downloaded(self):
        return os.path.exists(self.path)

    def download(self, overwrite: bool = False):

        if not self.is_downloaded or overwrite:

            # Create the target directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)

            # Call the function to extract the ZIP file
            extract_zip_from_url(url=self.url, target_dir=self.path)

            # Move content to power directory
            move_content_one_level_up(base_path=self.path)

            return

        print(f"File already downloaded at {self.path}")


def plot_cycles_comparison(results_a: Dict[str, Result], results_b: Dict[str, Result],
                           legend_a: str = "A", legend_b: str = "B", title: str = "Cycles comparison",
                           yrange=None, ax=None):
    cycles_a = [result.cycles for _, result in results_a.items()]
    cycles_b = [result.cycles for _, result in results_b.items()]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(range(len(cycles_a)), cycles_a, marker='o', color="blue",   label=legend_a)
    ax.plot(range(len(cycles_b)), cycles_b, marker='o', color="orange", label=legend_b)

    ax.axhline(y=mean(cycles_a), color="blue", linestyle='--', label=f'Mean {legend_a}')
    ax.axhline(y=mean(cycles_b), color="orange", linestyle='--', label=f'Mean {legend_b}')

    if yrange is not None:
        low, high = yrange
        ax.set_ylim(low, high)

    ax.set_xlabel('Word index')
    ax.set_ylabel('Cycles')
    ax.set_title(title)
    ax.legend()

    ax.set_xticks(range(1, len(cycles_a)+1, 3))
    ax.grid(True)


def print_list(l: List, title: str):

    print(title)
    for el in l:
        print(f"- {el}")

    print(f"Tot: {len(l)}")

