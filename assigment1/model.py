"""
pyDRC.py

Description:
This module defines classes related to the core concepts of language classification experiments,
including the meanings of words, individual words, collections of words, experimental subjects,
individual experiments, and a collection of experiments.

Classes:
- Meaning: Enum class representing the meanings "BIG" and "SMALL."
- Word: Data class representing a word with its properties such as id, word, language, meaning, and sound symbolism.
- Words: Class representing a collection of words with various methods for analysis and visualization.
- Subject: Class representing a subject (participant) in an experiment with methods to manage answers.
- Experiment: Class representing a language classification experiment for a specific subject,
  providing methods for analysis and visualization of results.
- Experiments: Class representing a collection of experiments, providing methods for analysis
  and visualization of experiment results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Iterable, List

import numpy as np
from matplotlib import pyplot as plt

from io_ import pie_plot, plot_confusion_matrix
from settings import VOWELS_COLORS, VOWELS, SMALL_BIG_COLORS, CORRECT_WRONG_COLORS, HISTO_COLORS


from enum import Enum


class Meaning(Enum):
    """
    Enumeration representing the meanings of words in a language classification model.
    """

    BIG = "big"
    SMALL = "small"

    def __str__(self) -> str:
        """
        Return the string representation of the meaning.

        :return: string representation of the meaning.
        """
        return self.value

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    @staticmethod
    def from_string(s: str) -> Meaning:
        """
        Convert a string to a Meaning enum value.

        :param s: string representation of the meaning.
        :return: meaning enum value.
        """

        if s == "small":
            return Meaning.SMALL

        if s == "big":
            return Meaning.BIG

        raise Exception(f"Invalid meaning string {s}, use one between {{'small', 'big'}}")

    @staticmethod
    def to_index(meaning: Meaning) -> int:
        """
        Return meaning to the corresponding index in a collection.

        :param meaning: meaning .
        :return: index.
        """

        return 0 if meaning == Meaning.SMALL else 1


@dataclass
class Word:
    """
    Data class representing a word in a language classification model.

    Attributes:
    - id_: Unique identifier for the word.
    - word: The actual word string.
    - language: Language of the word.
    - meaning: Meaning of the word, either 'big' or 'small'.
    - is_sound_symbolic: Indicates whether the word is sound symbolic or not.
    """

    id_: str
    word: str
    language: str
    meaning: Meaning
    is_sound_symbolic: bool

    def __str__(self):
        """
        Return a human-readable string representation of the word.

        :return: string representation of the word.
        """

        return f"{self.id_}[{self.word} ({self.language}); Meaning: {self.meaning}; " \
               f"{'' if self.is_sound_symbolic else 'not '}sound symbolic]"

    def __repr__(self):
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    def __len__(self):
        """
        Return the length of the word.

        :return: length of the word.
        """
        return len(self.word)

    @property
    def vowels_count(self) -> Dict[str, int]:
        """
        Return a dictionary with counts of each vowel in the word.

        :return: dictionary with counts of each vowel.
        """
        return {
            vowel: self.word.count(vowel)
            for vowel in VOWELS
        }


class Words:
    """
    Class representing a collection of words in a language classification model.
    """

    def __init__(self):
        """
        Create a new instance of Words.
        """

        self._words: Dict[str, Word] = dict()

    def __str__(self):
        """
        Return a human-readable string representation of the Words collection.

        :return: string representation of the Words collection.
        """

        n_small, n_big = self.get_meaning_count()
        return f"Words[count: {len(self)}, small: {n_small}, big: {n_big}]"

    def __repr__(self):
        """
        Return the string representation format.

        :return: string representation.
        """

        return str(self)

    def __getitem__(self, word_id: str) -> Word:
        """
        Get a Word instance by its ID.

        :param word_id: ID of the word to retrieve.
        :return: word instance.
        """

        return self._words[word_id]

    def __iter__(self):
        """
        Provide an iterator over the Word instances in the collection.

        :return: iterator over Word instances.
        """

        return iter(self._words.values())

    def __len__(self):
        """
        Return the number of words in the collection.

        :return: number of words.
        """

        return len(self._words)

    def add_word(self, word: Word):
        """
        Add a Word instance to the collection.

        :param word: Word instance to add.
        """

        self._words[word.id_] = word

    @property
    def avg_length(self) -> float:
        """
        Return the average length of words in the collection.

        :return: average length of words.
        """

        return sum([len(w) for w in self]) / len(self)

    @property
    def vowels_count(self) -> Dict[str, int]:
        """
        Return a dictionary with counts of each vowel in the collection.

        :return: dictionary with counts of each vowel.
        """

        return {
            vowel: sum([f"{word.word}".count(vowel) for word in self])
            for vowel in VOWELS
        }

    def plot_vowels_distr(self, title: str = "Vowels distribution", ax = None):
        """
        Plot the distribution of vowels in the collection.

        :param title: title for the plot.
        :param ax: Matplotlib Axes object for plotting.
        """

        pie_plot(
            labels=[s.upper() for s in VOWELS],
            sizes=list(self.vowels_count.values()),
            colors=VOWELS_COLORS,
            title=title,
            ax=ax
        )

    def get_meaning_count(self) -> Tuple[int, int]:
        """
        Return the count of words for each meaning category ('small' and 'big').

        :return: tuple containing the count of small words and big words.
        """

        n_small = len([word for word in self if word.meaning == Meaning.SMALL])
        n_big = len([word for word in self if word.meaning == Meaning.BIG])

        return n_small, n_big

    def get_meaning_split(self) -> Tuple[Words, Words]:
        """
        Split the collection into two based on meaning categories.

        :return: tuple containing two Words instances, one for 'small' words and one for 'big' words.
        """

        small_words, big_words = Words(), Words()

        # Add word to the two collection based on their meaning
        for word in self:
            if word.meaning == Meaning.SMALL:
                small_words.add_word(word=word)
            else:
                big_words.add_word(word=word)

        return small_words, big_words


class Subject:
    """
    Class representing a subject participating in a language classification experiment.
    """

    def __init__(self, id_: str):
        """
        Create a new instance of Subject.

        :param id_: unique identifier for the subject.
        """
        
        self._id: str = id_
        self._answers: Dict[str, Meaning] = dict()

    def __len__(self):
        """
        Return the number of answers provided by the subject.

        :return: number of answers.
        """
        
        return len(self._answers)

    def __str__(self):
        """
        Return a human-readable string representation of the subject.

        :return: string representation of the subject.
        """
        
        answer_small, answer_big = self.get_answer_count()
        return f"{self._id}[{answer_small} {Meaning.SMALL}, {answer_big} {Meaning.BIG}]"

    def __repr__(self):
        """
        Return the string representation format.

        :return: string representation.
        """
        
        return str(self)

    def __getitem__(self, word_id: str) -> Meaning:
        """
        Get the Meaning answer for a specific word ID.

        :param word_id: ID of the word to retrieve the answer for.
        :return: meaning answer.
        """
        
        return self._answers[word_id]

    def __iter__(self) -> Iterable[Tuple[str, Meaning]]:
        """
        Provide an iterator over word ID and Meaning pairs.

        :return: iterator over word ID and Meaning pairs.
        """

        return iter(self._answers.items())

    @property
    def id_(self) -> str:
        """
        Get the ID of the subject.

        :return: ID of the subject.
        """

        return self._id

    @property
    def small_word_ids(self) -> List[str]:
        """
        Get a list of word IDs with 'small' answers.

        :return: List of word IDs with 'small' answers.
        """

        return [word_id for word_id, meaning in self if meaning == Meaning.SMALL]

    @property
    def big_word_ids(self) -> List[str]:
        """
        Get a list of word IDs with 'big' answers.

        :return: list of word IDs with 'big' answers.
        """

        return [word_id for word_id, meaning in self if meaning == Meaning.BIG]

    def add_answer(self, word_id: str, answer: Meaning):
        """
        Add an answer for a specific word ID.

        :param word_id: ID of the word to add an answer for.
        :param answer: meaning answer.
        """

        self._answers[word_id] = answer

    def get_answer_count(self) -> Tuple[int, int]:
        """
        Get the count of 'small' and 'big' answers.

        :return: tuple containing the count of 'small' answers and 'big' answers.
        """

        return len(self.small_word_ids), len(self.big_word_ids)

    def plot_answer_count(self, title: str = "Answer distribution", ax=None):
        """
        Plot the distribution of 'small' and 'big' answers.

        :param title: title for the plot.
        :param ax: matplotlib Axes object for plotting.
        """

        answer_small, answer_big = self.get_answer_count()

        # Title includes subject-id
        pie_plot(
            labels=[str(meaning).upper() for meaning in [Meaning.SMALL, Meaning.BIG]],
            sizes=[answer_small, answer_big],
            colors=SMALL_BIG_COLORS,
            title=f"{self._id} {title}",
            ax=ax
        )


class Experiment:
    """
    Class representing a language classification experiment.
    """

    def __init__(self, subject: Subject, words: Words):
        """
        Initialize a new instance of Experiment.

        :param subject: subject participating in the experiment.
        :param words: set of words used in the experiment.
        """

        # Check matching between words-id
        word_id_sbj = set([word_id for word_id, _ in subject])
        word_id_wrd = set([word.id_ for word in words])

        if not word_id_sbj.issubset(word_id_wrd):
            raise Exception(
                f"Subject has some word-ids not present in the set of words {word_id_sbj.difference(word_id_wrd)}")

        self._subject = subject
        self._words = words

        # Evaluating results
        self._results: Dict[str, bool] = {
            word_id: self._subject[word_id] == self._words[word_id].meaning
            for word_id, _ in self.subject
        }

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the experiment.

        :return: string representation of the experiment.
        """

        return f"Experiment {self.subject.id_}[Score: {self.score}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """

        return str(self)

    def __len__(self) -> int:
        """
        Return the number of answers in the experiment.

        :return: number of results.
        """

        return len(self._results)

    def __iter__(self) -> Iterable[Tuple[str, bool]]:
        """
        Provide an iterator over word ID and answer.

        :return: iterator over word ID and answer.
        """

        return iter(self._results.items())

    def __getitem__(self, word_id: str) -> bool:
        """
        Get the correctness of the subject's answer for a specific word ID.

        :param word_id: ID of the word to retrieve the correctness for.
        :return: correctness of the subject's answer.
        """

        return self._results[word_id]

    @property
    def subject(self) -> Subject:
        """
        Get the subject participating in the experiment.

        :return: subject participating in the experiment.
        """

        return self._subject

    @property
    def words(self) -> Words:
        """
        Get the set of words used in the experiment.

        :return: set of words used in the experiment.
        """
        return self._words

    @property
    def results(self) -> Dict[str, bool]:
        """
        Get the dictionary of word IDs and correctness of subject's answers.

        :return: dictionary mapping word IDs to correctness of answers.
        """

        return self._results

    @property
    def words_correct(self) -> List[Word]:
        """
        Get a list of words for which the subject's answers were correct.

        :return: list of words for which the subject's answers were correct.
        """

        return [
            self.words[word_id] for word_id, correct in self if correct
        ]

    @property
    def words_wrong(self) -> List[Word]:
        """
        Get a list of words for which the subject's answers were wrong.

        :return: list of words for which the subject's answers were wrong.
        """
        return [
            self.words[word_id] for word_id, correct in self if not correct
        ]

    @property
    def n_correct(self) -> int:
        """
        Get the number of correct answers in the experiment.

        :return: number of correct answers.
        """
        return len(self.words_correct)

    @property
    def score(self) -> float:
        """
        Get the accuracy score of the subject's answers.

        :return: subject experiment score.
        """
        return self.n_correct / len(self)

    def plot_results(self, title: str = "Results", ax=None):
        """
        Plot the distribution of correct and wrong answers.

        :param title: title for the plot.
        :param ax: Matplotlib Axes object for plotting.
        """

        answer_small = self.n_correct
        answer_wrong = len(self) - self.n_correct

        pie_plot(
            labels=["Correct", "Wrong"],
            sizes=[answer_small, answer_wrong],
            colors=CORRECT_WRONG_COLORS,
            title=f"{self.subject.id_} - {title}",
            ax=ax
        )

    def plot_confusion_matrix(self, normalize_row: bool = False):

        # Compute CM
        cm = np.zeros((2, 2), dtype=int)

        # Updating prediction
        for word_id, answer in self.subject:

            # True
            true_label = Meaning.to_index(meaning=self.words[word_id].meaning)

            # Predicted
            predicted_label = Meaning.to_index(meaning=answer)

            # Updating matrix
            cm[true_label][predicted_label] += 1

        plot_confusion_matrix(
            cm=cm,
            normalize=normalize_row
        )


class Experiments:
    """
    Manages a collection of experiments, each associated with a subject, to evaluate word meaning classification.
    """

    def __init__(self, words: Words):
        """
        Initializes the Experiments instance with a collection of words.

        :param words: collection of words used in the experiments.
        """

        self._words: Words = words
        self._experiments: Dict[str, Experiment] = dict()

    def __str__(self) -> str:
        """
        Returns a string representation of the Experiments instance.

        :return: string representation.
        """

        return f"Experiment[Subjects: {len(self)} - Mean score: {round(self.mean_score, 3)}]"

    def __repr__(self) -> str:
        """
        Returns a string representation format.

        :return: string representation.
        """
        return str(self)

    def __len__(self):
        """
        Returns the number of experiments.

        :return: number of experiments.
        """
        return len(self._experiments)

    def __iter__(self) -> Iterable[Tuple[str, Experiment]]:
        """
        Provides an iterator over experiments indexes by the subject-id

        :return: iterator over experiments.
        """
        return iter(self._experiments.items())

    def __getitem__(self, subject_id: str) -> Experiment:
        """
        Retrieves an experiment by subject ID.

        :param subject_id: ID of the subject.
        :return: Associated experiment.
        """

        return self._experiments[subject_id]

    def add_subject(self, subject: Subject):
        """
        Adds a new subject to the experiments and conducts an associated experiment.

        :param subject: subject to add and experiment on.
        """

        new_experiment = Experiment(subject=subject, words=self._words)
        self._experiments[subject.id_] = new_experiment

    # SUBJECT

    @property
    def mean_score(self) -> float:
        """
        Computes the mean score across all experiments performed by different subjects.

        :return: mean experiment score.

        """
        return np.mean([experiment.score for _, experiment in self])

    def sort_by_score(self) -> List[Tuple[str, Experiment]]:
        """
        Sorts experiments by score in descending order.

        :return: sorted list of experiments by score.
        """

        experiments = list((subject_id, experiment) for subject_id, experiment in self)

        return sorted(experiments, key=lambda x: x[1].score, reverse=True)

    def plot_subjects_scores(self, n_row: int, by_score: bool = False):
        """
        Plots pie charts for each subject's scores.

        :param n_row: number of rows in the subplot grid.
        :param by_score: whether to sort subjects by score.
        """

        n_col = math.ceil(len(self) / n_row)
        fig, axes = plt.subplots(n_row, n_col, figsize=(10, 21))

        # If to sort by score
        iter_ = iter(self.sort_by_score()) if by_score else self

        # Plot
        for i, experiment in enumerate(iter_):
            _, experiment = experiment

            row = i // n_col
            col = i % n_col

            experiment.plot_results(ax=axes[row, col])

        fig.suptitle("Subjects score", fontsize=24)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # WORDS ANSWERS

    def get_word_answers(self) -> Dict[str, Tuple[int, int]]:
        """
        Retrieves the count of small and big answers for each word.

        :return: dictionary of word IDs with corresponding answer counts.
        """

        answers = dict()

        # Computing word answer count
        for word in self._words:

            n_small = len([_ for _, experiment in self if experiment.subject[word.id_] == Meaning.SMALL])
            n_big   = len([_ for _, experiment in self if experiment.subject[word.id_] == Meaning.BIG  ])

            answers[word.id_] = (n_small, n_big)

        # Sort
        answers = dict(sorted(answers.items(), key=lambda x: int(x[0][4:])))

        return answers

    def plot_words_answers(self, n_row: int):
        """
        Plots pie charts for the answers to each word.

        :param n_row: number of rows in the subplot grid.
        """

        n_col = len(self._words) // n_row
        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 21))

        for i, word_answers in enumerate(self.get_word_answers().items()):

            if i > len(self._words):
                break

            row = i // n_col
            col = i % n_col
            ax = axes[row][col]

            word_id, answers = word_answers
            word = self._words[word_id]

            pie_plot(
                labels=["Small", "Big"],
                sizes=list(answers),
                colors=SMALL_BIG_COLORS,
                title=f"{word.word}",
                ax=ax
            )

        fig.suptitle("Words answers", fontsize=24)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def get_evidence(self) -> Dict[str, List[Word]]:
        """
        Classifies words into evidence levels based on answer distribution.

        :return: Dictionary of evidence levels with corresponding words.
        """

        # Evidence Levels
        evidence = {level: [] for level in ["Strong", "High", "Medium", "Low", "None"]}

        for word_id, answers in self.get_word_answers().items():

            # Proportion
            evidence_level = max(answers) / sum(answers)

            word = self._words[word_id]

            # Switch among levels proportions
            if 0.5 <= evidence_level < 0.6:
                evidence["None"].append(word)
            elif 0.6 <= evidence_level < 0.7:
                evidence["Low"].append(word)
            elif 0.7 <= evidence_level < 0.8:
                evidence["Medium"].append(word)
            elif 0.8 <= evidence_level < 0.9:
                evidence["High"].append(word)
            else:
                evidence["Strong"].append(word)

        return evidence

    def plot_evidence(self):
        """
        Plots a histogram of evidence levels.
        """

        # Evidence count per level
        evidence_count = {k: len(v) for k, v in self.get_evidence().items()}

        # Plot a histogram
        labels = [f"{i*10} vs {100 - i*10}" for i in range(5, 10)]
        values = list(evidence_count.values())

        plt.bar(labels, values, color=HISTO_COLORS)
        plt.ylabel('Probability')
        plt.title('Evidence Level')
        plt.show()

    # RESULTS

    def get_words_results(self, sort: bool = False) -> List[Tuple[str, int]]:
        """
        Retrieves word results, optionally sorted.

        :param sort: whether to sort results.
        :return: list of word results.
        """

        # Count number of correct answers
        correct = [
            (word_id, answers[Meaning.to_index(self._words[word_id].meaning)])
            for word_id, answers in self.get_word_answers().items()
        ]

        # Sort
        if sort:
            correct = list(sorted(correct, key=lambda x: int(x[1]), reverse=True))

        return correct

    def plot_words_scores(self, n_row: int, sort: bool = False):
        """
        Plots pie charts for correct and wrong answers for each word.

        :param n_row: number of rows in the subplot grid.
        :param sort: whether to sort word results.
        """

        n_col = len(self._words) // n_row

        fig, axes = plt.subplots(n_row, n_col, figsize=(15, 20))

        for i, word_correct in enumerate(self.get_words_results(sort=sort)):

            if i > len(self._words):
                break

            row = i // n_col
            col = i % n_col
            ax = axes[row][col]

            word_id, correct = word_correct
            wrong = len(self) - correct
            word = self._words[word_id]

            pie_plot(
                labels=["Correct", "Wrong"],
                sizes=[correct, wrong],
                colors=CORRECT_WRONG_COLORS,
                title=f"{word.word} - {str(word.meaning).capitalize()}",
                ax=ax
            )

        fig.suptitle("Words scores", fontsize=24)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])


