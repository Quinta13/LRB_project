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
from matplotlib.axes import Axes

from io_ import extract_zip_from_url, move_content_one_level_up
from settings import DEFAULT_PARAMETER_PATH, COLS


# --- ENUM ---

class Named(Enum):
    """
    Enumeration representing named results.
    """

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
        """
        Return the color associated with the named result.

        :return: Color string associated with the named result.
        """

        colors = {
            Named.Correct: "green",
            Named.Wrong:   "red",
            Named.Lowac:   "black"
        }

        return colors[self]

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
        """
        Set a new default path for parameter files.

        :param new_path: The new default path for parameter files.
        """
        self.DEFAULT_PARAMETER_PATH = new_path

    @property
    def default(self) -> float:
        """
        Get the default value for the parameter.

        :return: The default value for the parameter.
        """
        return self.get_parameter_file_settings()[self]

    @staticmethod
    def get_parameter_file_settings(parameter_file: str | None = None) -> Dict[Parameter, float]:
        """
        Get the parameter settings from a parameter file.

        :param parameter_file: Path to the parameter file. If None, uses the default path.
        :return: Dictionary containing parameter settings.
        """
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

# --- RESULTS ---

@dataclass
class Result:
    """
    Class representing the result of a DRC simulation.
    """

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
        """
        Parse a line of results and create a Result instance.

        :param result_line: A line containing result information.
        :return: A Result instance.
        """

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
    """
    Class representing a set of results for different words with the same parameters from DRC simulations.
    """

    def __init__(self, results: List[Result]):
        """
        Initialize a ResultSet instance with a list of Result instances.

        :param results: List of Result instances.
        """

        self._results: Dict[str, Result] = {r.word: r for r in results}

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the result set.

        :return: String representation of the result set.
        """

        return f"ResultSet[{len(self)} results]"

    def __repr__(self):
        """
        Return the string representation format.

        :return: String representation.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return the number of results in the set.

        :return: Number of results.
        """

        return len(self._results)

    def __iter__(self) -> Iterable[Result]:
        """
        Iterate over the results in the set.

        :return: Iterator over Result instances.
        """

        return iter(self._results.values())

    def __getitem__(self, word: str) -> Result:
        """
        Get a result by word.

        :param word: Word to retrieve the result for.

        :return: Result instance.
        """

        return self._results[word]

    @property
    def avg_cycle(self) -> int:
        """
        Calculate the average cycle among all results.

        :return: Average cycle.
        """

        return int(mean([r.cycles for r in self]))
    
    @staticmethod
    def plot_cycles_comparison(results: List[ResultSet], legends: List[str],
                                title: str = "Cycles comparison",
                                yrange=None, figsize: Tuple[int, int] = (9, 4)):
        """
        Plot three types of cycles comparison for different result sets.

        :param results: List of ResultSet instances to compare.
        :param legends: List of legends for each result set.
        :param title: Title of the plot.
        :param yrange: Optional y-axis range.
        :param figsize: Option figure size for the plot.
        """
        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        ResultSet.plot_cycles_scatterplot(
            results=results,
            legends=legends,
            title=f"{title} - Scatterplot",
            ax=axes[0],
            yrange=yrange
        )

        #ResultSet.plot_cycles_boxplot(
        #    results=results,
        #    legends=legends,
        #    title=f"{title} - Boxplot",
        #    ax=axes[1]
        #)

        ResultSet.plot_cycles_violinplot(
            results=results,
            legends=legends,
            title=f"{title} - Violinplot",
            ax=axes[1]
        )
        
        

    @staticmethod
    def plot_cycles_scatterplot(results: List[ResultSet], legends: List[str],
                                title: str = "", figsize: Tuple[int, int] = (9, 4),
                                ax: Axes = None, yrange=None):
        """
        Plot a scatterplot of cycles for different result sets.

        :param results: List of ResultSet instances to compare.
        :param legends: List of legends for each result set.
        :param title: Title of the plot.
        :param yrange: Optional y-axis range.
        :param figsize: Option figure size for the plot.
        :param ax: Optional axes for the plot.
        """

        # Colors for each ResultSet

        if len(results) > len(COLS):
            raise Exception(f"Result list exceeding maximum number {len(COLS)}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        lines = []

        # Plot each ResultSet
        for result, col in zip(results, COLS[:(len(results))]):

            # Plot the two types of Named
            for marker, named in [('o', Named.Correct), ('x', Named.Wrong)]:
                i_cycle = [(i, r.cycles) for i, r in enumerate(result) if r.named == named]

                line = ax.scatter(
                    [i for i, _ in i_cycle],
                    [cycle for _, cycle in i_cycle],
                    marker=marker, color=col
                )
                lines.append(line)

                ax.axhline(y=result.avg_cycle, color=col, linestyle='--')

        # Set yrange from input
        if yrange is not None:
            low, high = yrange
            ax.set_ylim(low, high)

        # Graph settings
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

    @staticmethod
    def plot_cycles_boxplot(results: List[ResultSet], legends: List[str],
                            title: str = "Cycles boxplot", ax: Axes = None,
                            figsize: Tuple[int, int] = (9, 4)):
        """
        Plot a boxplot for cycles from different result sets.

        :param results: List of ResultSet instances to compare.
        :param legends: List of legends for each result set.
        :param title: Title of the plot.
        :param ax: Optional axes for the plot.
        :param figsize: Option figure size for the plot.
        """

        # Colors for each ResultSet

        if len(results) > len(COLS):
            raise Exception(f"Result list exceeding maximum number {len(COLS)}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        all_data = [[r.cycles for r in result] for result in results]

        # rectangular box plot
        bplot = ax.boxplot(all_data,
                           vert=True,
                           patch_artist=True,
                           labels=legends)
        ax.set_title('Rectangular box plot')

        for patch, color in zip(bplot['boxes'], COLS[:len(results)]):
            patch.set_facecolor(color)

        # Graph settings
        ax.set_ylabel('Cycles')
        ax.set_title(title)

    @staticmethod
    def plot_cycles_violinplot(results: List[ResultSet], legends: List[str],
                               title: str = "Cycles violinplot", ax: Axes = None,
                               figsize: Tuple[int, int] = (9, 4)) -> None:
        """
        Plot a violin plot for cycles from different result sets.

        :param results: List of ResultSet instances to compare.
        :param legends: List of legends for each result set.
        :param title: Title of the plot.
        :param ax: Optional axes for the plot.
        :param figsize: Option figure size for the plot.
        """

        if len(results) > len(COLS):
            raise Exception(f"Result list exceeding maximum number {len(COLS)}")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        all_data = [[r.cycles for r in result] for result in results]

        # Violin plot
        violin_parts = ax.violinplot(all_data, showmedians=True)

        # Set colors for each violin
        for partname in ('cbars', 'cmins', 'cmaxes'):
            vp = violin_parts[partname]
            vp.set_edgecolor('k')

        for pc, color in zip(violin_parts['bodies'], COLS[:len(results)]):
            pc.set_facecolor(color)

        # Set names on x-axis
        ax.set_xticks(range(1, len(legends) + 1))
        ax.set_xticklabels(legends, rotation='vertical', fontsize=7)

        # Graph settings
        ax.set_ylabel('Cycles')
        ax.set_title(title)


class Results:
    """
    Class representing a set of results for the same words varying a specific parameter from DRC simulations.
    """

    def __init__(self, results: Dict[float, Result], param: Parameter):
        """
        Initialize a Results instance with a dictionary of results and a parameter.

        :param results: Dictionary of results with parameter values as keys.
        :param param: Parameter associated with the results.
        """

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
        """
        Get a result by parameter value.

        :param value: Parameter value to retrieve the result for.

        :return: Result instance.
        """

        return self.results[value]

    @property
    def results(self) -> Dict[float, Result]:
        """
        Get the dictionary of results.

        :return: Dictionary of results.
        """

        return self._results

    @property
    def param(self) -> Parameter:
        """
        Get the varying parameter in the model.

        :return: Model varying parameter.
        """

        return self._param

    @property
    def word(self) -> str:
        """
        Get the word associated with the results.

        :return: Model word.
        """

        return list(self.results.values())[0].word

    def plot_cycles(self):
        """
        Plot the trend parameter-cycle for the model,
         with the additional information of pronounce and result for each point
        """

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

### --- DRC ---

class DRCDownloader:
    """
    Class for downloading the DRC software.
    """

    def __init__(self, path_: str, url: str):
        """
        Initialize a DRCDownloader instance with a local path and a download URL.

        :param path_: Local path where the downloaded files will be stored.
        :param url: URL from which to download the files.
        """

        self._path: str = path_
        self._url: str = url

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the downloader.

        :return: String representation of the downloader.
        """

        return f"DRCDownloader[path: {self.path}; url: {self.url}; {'' if self.is_downloaded else 'not'} downloaded]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: String representation.
        """

        return str(self)

    @property
    def path(self) -> str:
        """
        Get the local path where files are stored.

        :return: Local path.
        """

        return self._path

    @property
    def url(self) -> str:
        """
        Get the download URL.

        :return: Download URL.
        """

        return self._url

    @property
    def is_downloaded(self) -> bool:
        """
        Check if the files are already downloaded.

        :return: True if files are downloaded, False otherwise.
        """

        return os.path.exists(self.path)

    def download(self, overwrite: bool = False):
        """
        Download files from the specified URL to the local path.

        :param overwrite: If True, overwrite existing files. Default is False.
        """

        if not self.is_downloaded or overwrite:

            # Create the target directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)

            # Call the function to extract the ZIP file
            extract_zip_from_url(url=self.url, target_dir=self.path)

            # Move content to power directory
            move_content_one_level_up(base_path=self.path)

            return

        print(f"File already downloaded at {self.path}")

class DRCNetwork:
    """
    Class for managing DRC network configurations and run.
    """
    def __init__(self, dir_: str, binary: str):
        """
        Initialize a DRCNetwork instance with a directory and binary name.

        :param dir_: Directory where network software is stored.
        :param binary: Name of the associated DRC binary executable.
        """

        self._dir = dir_
        self._binary_name = binary

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the network.

        :return: String representation of the network.
        """
        return f"DRCNetwork[{os.path.join(self.dir_, self.binary)}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: String representation.
        """
        return str(self)

    @property
    def dir_(self) -> str:
        """
        Get the directory where network software are stored.

        :return: Directory path.
        """
        return self._dir

    @property
    def binary(self) -> str:
        """
        Get the name of the associated DRC binary executable.

        :return: Binary name.
        """
        return self._binary_name

    def clear(self):
        """
        Remove all DRC run histories from the directory.
        """
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
        self, word: str, parameters: List[Tuple[Parameter, float | Tuple[float, float, int]]] | str | None = None,
        files: bool = False, store_activations: bool = False, log: bool = False
    ) -> Result | ResultSet | Results | Tuple[Result, Activations] | Tuple[ResultSet, Activations] | Tuple[Results, Activations]:
        """
        Run the DRC program with the specified word and parameters.

        :param word: Word for simulation or TXT file containing a set of words.
        :param parameters: String indicating the parameter configuration file or
            list of tuples specifying parameters in format
            - (Parameter, Value) for changing the parameter;
            - (Parameter, (Start, End, Step) to perform step parameter simulations;
            Default is None indicating to use default parameters.
        :param files: Flag to indicate whether to include files in the output. Default is False.
        :param store_activations: Flag to indicate whether to store activations. Default is False.
        :param log: Flag to indicate whether to log the output. Default is False.

        :return: result or optionally a tuple containing activations.
        """

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

            try:
                result_line = output_lines.index("Results:") + 1
            except ValueError:
                raise Exception(f"Unable to parse Results:\n {output}")

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


class Activations:
    """
    Class for managing and visualizing DRC activations.
    """

    def __init__(self, activation_dir: str):
        """
        Initialize an Activations instance with an activation directory.

        :param activation_dir: Directory where activation files are stored.
        """

        self._dir: str = activation_dir

    def __str__(self) -> str:
        """
        Return the string representation format.

        :return: String representation.
        """

        return f"Activations[{self.dir_} - {len(self)} file{'s' if len(self)>1 else ''}]"

    def __repr__(self) -> str:
        """
        Get the number of activation files.

        :return: Number of activation files.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Get the number of activation files.

        :return: Number of activation files.
        """

        return len(self.file_names)

    def __iter__(self) -> Iterable[str]:
        """
        Iterate over the activation file names.

        :return: Iterator over activation file names.
        """

        return iter(self.file_names)

    @property
    def dir_(self) -> str:
        """
        Get the activation directory.

        :return: Activation directory path.
        """

        return self._dir

    @property
    def file_names(self) -> List[str]:
        """
        Get a list of activation file names in the directory.

        :return: List of activation file names.
        """

        return [file_ for file_ in os.listdir(self.dir_) if file_.endswith(".acts")]

    @staticmethod
    def _parse_file(activation_file: str) -> Dict[str, List[float]]:
        """
        Parse the contents of an activation file.

        :param activation_file: Path to the activation file.
        :return: Dictionary containing activation data.
        """

        def extract_float(line_: str) -> float:
            """
            Extract the float token from a line
            """
            for token in line_.split():
                try:
                    float_token = float(token)
                    return float_token
                except ValueError:
                    pass

        # Activation types
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
        """
        Get activation data from a specific file.

        :param file_name: Name of the activation file.
        :return: Dictionary containing activation data.
        """

        activation_file = path.join(self.dir_, file_name)

        return self._parse_file(activation_file=activation_file)

    def plot(self, file_name: str, main: str | None = None,  ax: Axes = None):
        """
        Plot activation data from a specific file.

        :param file_name: Name of the activation file.
        :param main: Main title for the plot. Default is None.
        :param ax: Matplotlib axis for the plot. Default is None.
        """

        if ax is None:
            _, ax = plt.subplots()

        if main is None:
            main = file_name

        for a_name, a_value in self.get_data(file_name=file_name).items():
            ax.plot(range(len(a_value)), a_value, label=a_name)

        ax.set_title(main)
        ax.legend(prop={"size": 6})

    def plot_multiple(self, nrows: int = 2, figsize: Tuple[int, int] = (15, 12), order: List[int] | None = None):
        """
        Plot activations from multiple files in a grid layout.

        :param nrows: Number of rows in the grid. Default is 2.
        :param figsize: Size of the entire figure. Default is (15, 12).
        :param order: Order in which to plot activation files. Default is None.
        """

        ncols = math.ceil(len(self) / nrows)

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
