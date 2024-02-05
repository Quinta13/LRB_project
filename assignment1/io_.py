"""
Filename: io_.py

Description:
This file provides classes for handling data download and loading operations.

Classes:
1. DataDownloader:
   - Downloads a file from a specified URL and stores it locally.
   - Allows customization of the target directory and file name.
   - Provides methods for checking if the resource is already downloaded and downloading the resource.

2. CSVLoader:
   - Loads data from a CSV file into a pandas DataFrame.
   - Provides methods for accessing the file path and loading the data.
   - Supports loading the data only once.

3. TXTLoader:
   - Loads data from a text file into a list of strings.
   - Provides methods for accessing the file path and loading the data.
   - Supports loading the data only once.
"""


import os
import urllib.request
from typing import Callable, List

import numpy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class DataDownloader:
    """Downloads a file from a given url and store it locally"""

    def __init__(self, url: str, dir_path: str, file_name: str):
        """
        Create a new instance of DataDownloader.

        :param url: resource location.
        :param dir_path: directory where store the file.
        :param file_name: file to save the resource (including extension).
        """

        self._url: str = url
        self._dir_path: str = dir_path
        self._file_name: str = file_name

    def __str__(self) -> str:
        """
        Return the string representation in a human-readable format.
        :return: string representation.
        """
        return f"DataDownloader[file: {self.file_path}; downloaded: {self.is_downloaded}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.
        :return: string representation.
        """
        return str(self)

    @property
    def url(self) -> str:
        """
        Return resource location.
        :return: resource location.
        """
        return self._url

    @property
    def dir_path(self) -> str:
        """
        Return directory path.
        :return: directory path.
        """
        return self._dir_path

    @property
    def file_name(self) -> str:
        """
        Return file name.
        :return: file name.
        """
        return self._file_name

    @property
    def file_path(self) -> str:
        """
        Return complete file path where to store the resource.
        :return: resource file path.
        """
        return os.path.join(self.dir_path, self.file_name)

    @property
    def is_downloaded(self) -> bool:
        """
        Return if the resource was already downloaded.
        :return: is the resource is already downloaded.
        """
        return os.path.exists(self.file_path)

    def download(self, replace: bool = False):
        """
        Download the resource to target file path.
        :param replace: if to download the resource even if already present;
                        the older one will be overwritten.
        """

        # Create the directory if it doesn't exist
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
            print(f"Directory created: {self.dir_path}")
        else:
            print(f"Directory already exists: {self.dir_path}")

        # Download condition
        if not self.is_downloaded or replace:

            # Download the file from the URL
            try:
                urllib.request.urlretrieve(self.url, self.file_path)
                print(f"File downloaded successfully: {self.file_path}")
            except Exception as e:
                print(f"Error downloading file: {e}")

        print(f"File downloaded at: {self.file_path}")


class CSVLoader:
    """Loads data from a CSV file, saves the resource path, and reads only once."""

    def __init__(self, file_path: str):
        """
        Create a new instance of CSVLoader.

        :param file_path: Path to the CSV file.
        """

        self._file_path: str = file_path
        self._loader: Callable = pd.read_csv
        self._loaded: bool = False
        self._data: pd.DataFrame | None = None

    def __str__(self) -> str:
        """
        Return the string representation in a human-readable format.

        :return: string representation.
        """
        return f"CSVLoader[file: {self._file_path}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    @property
    def file_path(self) -> str:
        """
        Return the path to the CSV file.

        :return: file path.
        """
        return self._file_path

    def load(self) -> pd.DataFrame:
        """
        Load data from the CSV file.
        If the data has already been loaded, it returns the cached DataFrame.

        :return: loaded DataFrame.
        """
        if self._loaded:
            return self._data

        print(f"Loading {self.file_path} ...")

        self._data = pd.read_csv(filepath_or_buffer=self.file_path)
        self._loaded = True

        print(f"Complete")

        return self._data


class TXTLoader:
    """Loads data from a text file and saves the resource path."""

    def __init__(self, file_path: str):
        """
        Create a new instance of TXTLoader.

        :param file_path: path to the text file.
        """
        self._file_path: str = file_path
        self._loaded: bool = False
        self._data: List[str] = []

    def __str__(self) -> str:
        """
        Return the string representation in a human-readable format.

        :return: string representation.
        """
        return f"TXTLoader[file: {self._file_path}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    @property
    def file_path(self) -> str:
        """
        Return the path to the text file.

        :return: file path.
        """
        return self._file_path

    def load(self) -> List[str]:
        """
        Load data from the text file.
        If the data has already been loaded, it returns the cached list of strings.

        :return: loaded list of strings.
        """
        if self._loaded:
            return self._data

        try:
            with open(self.file_path, 'r') as file:
                self._data = [line.rstrip('\n') for line in file]
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")

        self._loaded = True
        return self._data


# PLOT
def pie_plot(labels: List[str], sizes: List[int | float], colors: List[str],
             title: str = "", ax=None):
    """
    Create a pie chart plot.

    :param labels: list of labels for each pie segment.
    :param sizes: list of sizes or proportions for each pie segment.
    :param colors: list of colors for each pie segment.
    :param title: title of the pie chart (default is an empty string).
    :param ax: axes object for the plot. If None, a new subplot is created.
    :return: axes object with the pie chart plot.
    """

    # Use single plot if not axes is specified
    if ax is None:
        _, ax = plt.subplots()

    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.set_title(title)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


def plot_confusion_matrix(cm: np.ndarray, normalize: bool = False):
    """
    Plot given confusion matrix.
    :param cm: 2x2 confusion matrix.
    :param normalize: if to normalize rows turning count to probability.
    """

    # Row normalization
    if normalize:
        cm = cm.astype(float)
        for i in [0, 1]:
            cm[i] = cm[i] / sum(cm[i])

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Confusion Matrix')
    plt.colorbar()

    class_names = ["Small", "Big"]
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, str(round(cm[i, j], 2)), horizontalalignment="center", color="black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()