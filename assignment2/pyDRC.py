from __future__ import annotations

import itertools
import math
import os.path
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from os import path
from statistics import mean
from typing import Dict, List, Tuple, Iterable

from matplotlib import pyplot as plt

from io_ import extract_zip_from_url, move_content_one_level_up
from settings import DEFAULT_PARAMETER_PATH

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

class Named(Enum):

    Correct = "NAMED CORRECT"
    Wrong = "NAMED WRONG"
    Lowac = "LOWAC WRONG"

    def __str__(self) -> str:
        """
        Return the string representation of the named result.

        :return: string representation of the named result.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    def to_color(self) -> str:

        if self == Named.Correct:
            return "green"
        elif self == Named.Wrong:
            return "red"
        else:
            return "black"

class Parameter(Enum):
    """
    Enumeration representing the parameter of the model.
    """

    # Default parameter path
    DEFAULT_PARAMETER_PATH = ""

    # General Parameters
    ActivationRate = "ActivationRate"
    FrequencyScale = "FrequencyScale"
    MinReadingPhonology = "MinReadingPhonology"

    # Feature Level Parameters
    FeatureLetterExcitation = "FeatureLetterExcitation"
    FeatureLetterInhibition = "FeatureLetterInhibition"

    # Letter Level Parameters
    LetterOrthlexExcitation = "LetterOrthlexExcitation"
    LetterOrthlexInhibition = "LetterOrthlexInhibition"
    LetterLateralInhibition = "LetterLateralInhibition"

    # Orthographic Lexicon (Orthlex) Parameters
    OrthlexPhonlexExcitation = "OrthlexPhonlexExcitation"
    OrthlexPhonlexInhibition = "OrthlexPhonlexInhibition"
    OrthlexLetterExcitation = "OrthlexLetterExcitation"
    OrthlexLetterInhibition = "OrthlexLetterInhibition"
    OrthlexLateralInhibition = "OrthlexLateralInhibition"

    # Phonological Lexicon (Phonlex) Parameters
    PhonlexPhonemeExcitation = "PhonlexPhonemeExcitation"
    PhonlexPhonemeInhibition = "PhonlexPhonemeInhibition"
    PhonlexOrthlexExcitation = "PhonlexOrthlexExcitation"
    PhonlexOrthlexInhibition = "PhonlexOrthlexInhibition"
    PhonlexLateralInhibition = "PhonlexLateralInhibition"

    # Phoneme Level Parameters
    PhonemePhonlexExcitation = "PhonemePhonlexExcitation"
    PhonemePhonlexInhibition = "PhonemePhonlexInhibition"
    PhonemeLateralInhibition = "PhonemeLateralInhibition"
    PhonemeUnsupportedDecay = "PhonemeUnsupportedDecay"

    # GPC Route Parameters
    GPCPhonemeExcitation = "GPCPhonemeExcitation"
    GPCCriticalPhonology = "GPCCriticalPhonology"
    GPCOnset = "GPCOnset"

    def __str__(self) -> str:
        """
        Return the string representation of the parameter.

        :return: string representation of the parameter.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    def set_default_parameter_path(self, new_path: str) -> None:
        self.DEFAULT_PARAMETER_PATH = new_path

    @property
    def default(self) -> float:
        return self.get_parameter_file_settings()[self]

    @staticmethod
    def get_parameter_file_settings(parameter_file: str | None = None) -> Dict[Parameter, float]:

        if parameter_file is None:
            parameter_file = DEFAULT_PARAMETER_PATH

        with open(parameter_file, 'r') as file:
            file_content = file.read()

        # Filtering out non-param rows; splitting between non params
        params = {
            Parameter(line.split()[0]): float(line.split()[1]) for line in file_content.splitlines()
            if line.strip() != "" and not line.startswith('#')
        }

        return params


class DRCNetwork:

    def __init__(self, dir_: str, binary: str):

        self._dir = dir_
        self._binary_name = binary

    def __str__(self) -> str:
        return f"DRCNetwork[{os.path.join(self.dir_, self.binary)}]"

    def __repr__(self) -> str:
        return str(self)

    @property
    def dir_(self) -> str:
        return self._dir

    @property
    def binary(self) -> str:
        return self._binary_name

    def clear(self):

        # Get a list of all items in the folder
        items = os.listdir(self.dir_)

        # Iterate through items in the folder
        for item in items:
            item_path = os.path.join(self.dir_, item)

            # Check if the item is a directory and ends with ".drc"
            if os.path.isdir(item_path) and item.endswith(".drc"):
                # Use shutil.rmtree to remove the directory and its contents
                shutil.rmtree(item_path)
                print(f"Removed directory: {item_path}")


    def run(
        self, word: str, parameters: List[Tuple[Parameter, float | tuple]] | str | None = None,
        files: bool = False, store_activations: bool = False, log: bool = False
    ) -> Result | ResultSet | Results | Tuple[Result, Activations] | Tuple[ResultSet, Activations] | Tuple[Results, Activations]:

        # We need to change working directory to drc program
        # We save old one, and we reset it at the end of the process
        work_dir = os.getcwd()
        os.chdir(self.dir_)

        # Flags to discriminate one result vs. multiple results
        multiple_result = False
        param = None

        # Arguments to be passed to the binary
        activation_file=""

        # Store activation option
        if store_activations:

            # If activation are stored, also files are
            files = True

            word_ = word.split(".")[0] if word.endswith(".txt") else word

            # Count how many files exists yet with word name
            count = len([
                file for file in os.listdir(self.dir_)
                 if file.startswith(f"{word_}.drc") or file.startswith(f"{word_}-")
            ])

            # Activation file path
            activation_dir = path.join(
                self.dir_,
                f"{word_}.drc" if count == 0 else f"{word_}-{count}.drc",  # cope with multiple version of the same word
            )

            activation = Activations(activation_dir=activation_dir)

        files_arg = ([] if files else ['--nofiles']) + (['-a'] if store_activations else [])

        # Params
        params = []

        # Parameters from file
        if type(parameters) == str:

            params += ["-p", parameters]

        # Parameters specified by command line
        else:

            if parameters is None: # No parameter specified
                parameters = []

            for param, value in parameters:
                if type(value) == tuple:
                    # Parameter step
                    multiple_result = True
                    params += ["-S"] + [f"{param}"] + list(value)
                else:
                    # Parameter value set
                    params += ["-P"] + [f"{param}"] + [value]

        # Single words or from file
        words = ['-b', word] if word.endswith(".txt") else [word]

        # Combine arguments and cast numeric to strings
        args = files_arg + params + words
        args = [str(a) for a in args]

        # Use subprocess to run the binary with the argument and capture its output
        try:

            command = [self.binary] + args
            print(f"Running: {'./' + ' '.join(command)}")

            # Run the binary with subprocess.check_output
            output = subprocess.check_output(
                command,
                stderr=subprocess.STDOUT,
                text=True
            )

            if log:
                print(output)

            # Reset working directory
            os.chdir(work_dir)

            # Find the index of the line containing "Results:"
            output_lines = output.split('\n')

            result_line = output_lines.index("Results:") + 1

            if not multiple_result:

                # We expect only one result

                # Single word
                if word.endswith('.txt'):

                    # Extract result lines
                    result_lines = output_lines[result_line:]
                    result_lines = result_lines[:result_lines.index("")]

                    result = ResultSet(results=[
                        Result.parse_results_line(result_line=line) for line in result_lines
                    ])

                # Multiple words
                else:
                    result = Result.parse_results_line(result_line=output_lines[result_line])

                # Different type of output depending on activation
                if not store_activations:
                    return result
                else:
                    return result, activation

            else:

                # Extract lines containing results
                result_lines = output_lines[result_line:]
                result_lines = result_lines[:result_lines.index("")]

                results_dict = {
                    float(param_line.split()[1]): Result.parse_results_line(result_line=result_line)
                    for param_line, result_line in zip(result_lines[1::3], result_lines[2::3])
                }

                results = Results(
                    results=results_dict,
                    param=param
                )

                if not store_activations:
                    return results
                else:
                    return results, activation

        except subprocess.CalledProcessError as e:

            # Handle any errors that might occur during execution
            print("Error:", e.output)
            os.chdir(work_dir)


@dataclass
class Result:

    word: str
    pronounce: str
    cycles: int
    named: Named

    def __str__(self):
        """
        Return a human-readable string representation of the result.

        :return: string representation of the result.
        """

        return f"[{self.word}: {self.pronounce} - {self.named}; cycles: {self.cycles}]"

    def __repr__(self):
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    @staticmethod
    def parse_results_line(result_line: str) -> Result:

        result_data = result_line.split()

        # Cope with the case in which a space is given as pronunciation
        if result_line[len(result_data[0])+1] == " ":

            word, cycles, named = tuple(result_data[:2] + [" ".join(result_data[2:])])
            pronounce = " "

        else:
            # Create a tuple with the desired information
            word, pronounce, cycles, named = tuple(result_data[:3] + [" ".join(result_data[3:])])

        return Result(
            word=word,
            pronounce=pronounce,
            cycles=int(cycles),
            named=Named(named)
        )

class ResultSet:

    def __init__(self, results: List[Result]):

        self._result: Dict[str, Result] = {r.word: r for r in results}

    def __str__(self) -> str:
        return f"ResultSet[{len(self)} results]"

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return len(self._result)

    def __iter__(self):
        return iter(self._result.values())

    def __getitem__(self, word: str) -> Result:
        return self._result[word]

    @property
    def avg_cycle(self) -> int:
        return int(mean([r.cycles for r in self]))

    @staticmethod
    def plot_cycles_comparison(results: List[ResultSet], legends: List[str],
                               title: str = "Cycles comparison", yrange=None, ax=None):

        COLS = ['blue', 'orange', 'green', 'purple']  # add more

        if len(results) > len(COLS):
            raise Exception(f"Result list exceeding maximum number {len(COLS)}")

        if ax is None:
            fig, ax = plt.subplots()

        lines = []

        for result, col in zip(results, COLS[:(len(results))]):

            for marker, named in [('o', Named.Correct), ('x', Named.Wrong)]:
                i_cycle = [(i, r.cycles) for i, r in enumerate(result) if r.named == named]

                line = ax.scatter(
                    [i for i, _ in i_cycle],
                    [cycle for _, cycle in i_cycle],
                    marker=marker, color=col
                )
                lines.append(line)

                ax.axhline(y=result.avg_cycle, color=col, linestyle='--')

        if yrange is not None:
            low, high = yrange
            ax.set_ylim(low, high)

        ax.set_xlabel('Word index')
        ax.set_ylabel('Cycles')
        ax.set_title(title)

        max_range = max([len(r) for r in results])
        ax.set_xticks(range(1, max_range + 1, 4))
        ax.grid(True)

        # Add legend for markers
        for legend, color in zip(legends, COLS[:len(legends)]):
            ax.scatter([], [], marker='o', color=color, label=legend)
        ax.scatter([], [], marker='o', color='black', label=str(Named.Correct))
        ax.scatter([], [], marker='x', color='black', label=str(Named.Wrong))
        ax.plot([], linestyle='--', color='black', label='Average')
        ax.legend(loc='upper left', prop={"size": 6})

class Results:

    def __init__(self, results: Dict[float, Result], param: Parameter):

        self._results = results
        self._param = param

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result.

        :return: string representation of the result.
        """

        param_values = list(self.results.keys())

        return f"Results[{self.word} - {self.param}: {{from: {param_values[0]}, to: {param_values[-1]}, step: {len(self)}}}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    def __len__(self) -> int:
        """
        Return the considered number of results.

        :return: number of results.
        """
        return len(self.results)

    def __getitem__(self, value) -> Result:
        return self.results[value]

    @property
    def results(self) -> Dict[float, Result]:
        return self._results

    @property
    def param(self) -> Parameter:
        return self._param

    @property
    def word(self) -> str:
        return list(self.results.values())[0].word

    def plot_cycles(self):
        import matplotlib.pyplot as plt

        # Data
        x          = list(self.results.keys())
        y          = [v.cycles            for v in self.results.values()]
        colors     = [v.named.to_color() for v in self.results.values()]
        pronounces = [v.pronounce         for v in self.results.values()]

        # Create a scatter plot for each label and color
        plt.scatter(x, y, c=colors, marker='.')

        # Connect the points with lines

        for x_, y_, color, pronounce in zip(x, y, colors, pronounces):
            plt.text(
                x_, y_, pronounce, fontsize=8, color='black', ha='left', va='bottom'
            )

        plt.plot(x, y, color='gray', linestyle='-', linewidth=1)

        # Add labels to the axes
        plt.xlabel(f'{self.param}')
        plt.ylabel('Cycles')

        # Add a title to the plot
        plt.title(f'{self.word} cycles varying {self.param}')

        # Legend
        legend_dict = {named.to_color(): str(named) for named in [Named.Correct, Named.Wrong, Named.Lowac]}

        custom_handles = [
            plt.Line2D([0], [0], marker='.', color='w', markerfacecolor=color, markersize=10)
            for color in legend_dict.keys()
        ]

        custom_labels = list(legend_dict.values())

        plt.legend(
            handles=custom_handles,
            labels=custom_labels,
            prop={"size": 6}
        )

        # Plot
        plt.show()


class Activations:

    def __init__(self, activation_dir: str):

        self._dir: str = activation_dir

    def __str__(self) -> str:

        return f"Activations[{self.dir_} - {len(self)} file{'s' if len(self)>1 else ''}]"

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.file_names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.file_names)

    @property
    def dir_(self) -> str:
        return self._dir

    @property
    def file_names(self) -> List[str]:
        return [file_ for file_ in os.listdir(self.dir_) if file_.endswith(".acts")]

    @staticmethod
    def _parse_file(activation_file: str) -> Dict[str, List[float]]:

        def extract_float(line_: str) -> float:

            for token in line_.split():
                try:
                    float_token = float(token)
                    return float_token
                except ValueError:
                    pass

        activations = {
            "TL": [],
            "TGPC": [],
            "TPh": [],
            "TP": [],
            "TGPCR": []
        }

        activations_checks = ["TGPC ", "TP ", "TPh", "TGPCR"]

        cycles = 0

        with open(activation_file, "r") as f:

            for line in f:

                float_line = extract_float(line_=line)

                if "TL" in line:

                    cycles += 1

                    activations["TL"].append(float_line)

                    for a in activations_checks:

                        if len(activations[a.strip()]) == cycles - 2:
                            activations[a.strip()].append(0)

                else:
                    for a in activations_checks:
                        if a in line:
                            activations[a.strip()].append(float_line)

        return activations

    def get_data(self, file_name: str) -> Dict[str, List[float]]:

        activation_file = path.join(self.dir_, file_name)

        return self._parse_file(activation_file=activation_file)

    def plot(self, file_name: str, main: str | None = None,  ax=None):

        if ax is None:
            _, ax = plt.subplots()

        if main is None:
            main = file_name

        for a_name, a_value in self.get_data(file_name=file_name).items():
            ax.plot(range(len(a_value)), a_value, label=a_name)

        ax.set_title(main)
        ax.legend(prop={"size": 6})

    def plot_multiple(self, nrows: int = 2, figsize: Tuple[int, int] = (15, 12), order: List[int] | None = None):

        ncols = math.ceil(len(self) / (nrows))

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if order is None:
            files_iter = iter(self)

        else:
            files_iter = iter([self.file_names[i] for i in order])

        for i, j in itertools.product(range(nrows), range(ncols)):

            try:
                file_name = next(files_iter)
                self.plot(file_name=file_name, ax=axes[i][j])
            except StopIteration: # case less activations than grid cells
                break