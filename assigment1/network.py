"""
Filename: model.py

Description:
This file contains the implementation of classes related to a meaning classification model.

Classes:
1. Alphabet:
   - Represents an alphabet with mappings for characters to integer indices.
   - Provides methods for adding words to the alphabet and converting words to tensors.

2. MeaningClassifier:
   - A neural network model for meaning classification.
   - Utilizes an alphabet for word encoding and implements methods for prediction.

3. ModelConfig:
   - Data class representing configuration parameters for a language classification model.
   - Includes attributes for the number of training epochs, logging intervals, learning rate, and device.

4. NetworkAnalysis:
   - Performs analysis and visualizations for a neural network model.
   - Includes methods for plotting training loss, predicting language probabilities, plotting confusion matrix,
     and visualizing embeddings using t-SNE.

5. Trainer:
   - Manages the training process for a language classification model.
   - Utilizes a specified model, dataset of words, and configuration parameters for training.
   - Provides methods for generating training instances, training the model, and logging training progress.

"""

import random
from dataclasses import dataclass
from typing import Set, Dict, Iterable, Tuple, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn, Tensor, optim
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from io_ import plot_confusion_matrix
from model import Words, Word, Meaning
from settings import SMALL_BIG_COLORS


class Alphabet:
    """Represents an alphabet with mappings for characters to integer indices."""

    def __init__(self):
        """Create a new instance of the Alphabet class."""

        self._alphabet: Set[str] = set()
        self._alphabet_mapping: Dict[str, int] = dict()

    def __str__(self) -> str:
        """
        Return the string representation in a human-readable format.

        :return: string representation.
        """

        return f"Alphabet[size: {len(self)}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """
        return str(self)

    def __len__(self) -> int:
        """
        Return the size of the alphabet.

        :return: size of the alphabet.
        """
        return len(self._alphabet)

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        """
        Return an iterator over the alphabet mapping.

        :return: iterator over alphabet mapping as (character, index) tuples.
        """

        return iter(self._alphabet_mapping.items())

    def __getitem__(self, character: str) -> int:
        """
        Get the index of a character in the alphabet.

        :param character: Character to get the index for.
        :return: index of the character in the alphabet.
        """

        return self._alphabet_mapping[character]

    @property
    def alphabet_mapping(self) -> Dict[str, int]:
        """
        Return the alphabet mapping.

        :return: dictionary mapping characters to their indices in the alphabet.
        """

        return self._alphabet_mapping

    def add_word(self, word: str):
        """
        Add characters from a word to the alphabet.

        :param word: word to add to the alphabet.
        """

        # Loop on all word characters
        for char in set(word):

            # Check if already present
            if char not in self._alphabet:
                # Adding to the char-set and to the mapping
                self._alphabet_mapping[char] = len(self)
                self._alphabet.add(char)

    def word_to_tensor(self, word: str) -> Tensor:
        """
        Convert a word to a tensor of character indices.

        :param word: word to convert to a tensor.
        :return: tensor representing the word as character indices.
        """

        tensor_list = []

        # The tensor has as many rows as chars in the list
        for char in word:
            char_idx = self.alphabet_mapping[char]
            tensor_list.append([char_idx])

        # Converting to a tensor for training
        tensor = torch.tensor(tensor_list)

        return tensor


class MeaningClassifier(nn.Module):

    def __init__(self, alphabet: Alphabet, hidden_size: int):
        """
        Initializes a LanguageClassifier instance.

        :param alphabet: alphabet used for encoding input.
        :param hidden_size: size of the hidden state in the model.
        """

        super().__init__()

        self.alphabet: Alphabet = alphabet

        # Model hyper-parameters
        self.alphabet_size: int = len(self.alphabet)
        self.hidden_size: int = hidden_size
        self.meanings: int = 2  # Small and Big

        # Layers
        self.input_to_hidden = nn.Embedding(self.alphabet_size, self.hidden_size // 2)
        self.hidden_to_hidden = nn.Linear(self.hidden_size, self.hidden_size // 2)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.meanings)
        self.softmax = nn.LogSoftmax(dim=1)

    def get_starting_hidden(self) -> Tensor:
        """
        Get the initial hidden state of the model.

        :return: initial hidden state.
        """

        return torch.zeros(1, self.hidden_size)

    def forward(self, input_: Tensor, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform a forward step in the network by combining actual input and previous hidden state

        :param input_: input tensor in the form of an embedding entry.
        :param hidden: the tensor describing the hidden state compute at the previous step of recursion.
        :return: the probability distribution over languages and the actual hidden state.
        """

        # Input to hidden
        embedded_input = self.input_to_hidden(input_)
        hidden_new = self.hidden_to_hidden(hidden)

        # Combine input and hidden transformation
        hidden_concat = torch.cat((embedded_input, hidden_new), 1)

        # Hidden to output
        output = self.hidden_to_output(hidden_concat)
        output_probs = self.softmax(output)

        return output_probs, hidden_concat


@dataclass
class ModelConfig:
    """
    Data class representing configuration parameters for a model.

    Attributes:
    - epochs: number of training epochs.
    - epochs_log: interval for logging during training.
    - lr: learning rate for the model.
    - device: device on which to train the model.
    """

    epochs: int
    epochs_log: int
    lr: float
    device: torch.device

    def __str__(self) -> str:
        """
        Return the string representation in a human-readable format.

        :return: string representation.
        """

        return f"ModelConfig[epochs: {self.epochs}; epochs-log: {self.epochs_log}; lr: {self.lr}; device: {self.device}]"

    def __repr__(self) -> str:
        """
        Return the string representation format.

        :return: string representation.
        """

        return str(self)


class NetworkAnalysis:
    """
    Performs analysis and visualizations for a neural network model.
    """

    def __init__(self, model: nn.Module, losses: List[float]):
        """
        Create a new instance of NetworkAnalysis.

        :param model: neural network model.
        :param losses: list of training losses over epochs.
        """

        self._model: nn.Module = model
        self._losses: List[float] = losses

    def plot_loss(self, downsampling_factor: int = 500, title: str = ""):
        """
        Plot the training loss over epochs with optional downsampling.

        :param downsampling_factor: specification of the downsampling factor.
        :param title: string representing the plot title.
        """

        # Points per bin
        points_per_bin = len(self._losses) // downsampling_factor

        # Downsampling by taking bin mean
        downsampled_data = [
            np.mean(self._losses[i:i + points_per_bin])
            for i in range(0, len(self._losses), points_per_bin)
        ]

        # Ticks
        x_ticks = [i * points_per_bin for i, _ in enumerate(downsampled_data)]

        # Plot
        plt.plot(x_ticks, downsampled_data)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()

    def predict(self, word: str) -> Tuple[float, float]:
        """
        Get the probabilities predictions for a given surname using the model.

        :param word: word for language prediction.
        :return: prediction probabilities.
        """

        # Tensors
        word_tensor = self._model.alphabet.word_to_tensor(word=word)
        hidden = self._model.get_starting_hidden()

        # Evaluate prediction
        for word_row in word_tensor:
            output, hidden = self._model(word_row, hidden)

        # Turning to probabilities
        probabilities = F.softmax(output, dim=1)

        p_small = probabilities[0, 0].item()
        p_big = probabilities[0, 1].item()

        return p_small, p_big

    def plot_confusion_matrix(self, words: Words, normalize_row: bool = False):
        """
        Plot the confusion matrix for the given dataset.

        :param words: words for confusion matrix evaluation.
        :param normalize_row: if to normalize rows.
        """

        # Compute CM
        cm = np.zeros((2, 2), dtype=int)

        # Updating prediction
        for word in words:

            # True
            true_label = Meaning.to_index(meaning=word.meaning)

            # Predicted
            p_small, p_big = self.predict(word=word.word)
            predicted_label = 0 if p_small >= p_big else 1

            # Updating matrix
            cm[true_label][predicted_label] += 1

        plot_confusion_matrix(
            cm=cm,
            normalize=normalize_row
        )

    def plot_embedding(self, words: Words):
        """
        Plot the embeddings  using a t-SNE visualization.

        :param words: list of words for embedding visualization.
        """

        # Perform dimensionality reduction
        tsne = TSNE(n_components=2)

        # Computing embeddings
        hiddens = []
        labels = []

        for i, word in enumerate(words, start=1):

            # Tensors
            word_tensor = self._model.alphabet.word_to_tensor(word=word.word)
            hidden = self._model.get_starting_hidden()

            # Evaluate prediction
            for word_row in word_tensor:
                output, hidden = self._model(word_row, hidden)

            hiddens.append(hidden.detach().numpy())
            labels.append(Meaning.to_index(meaning=word.meaning))

        # Stacking embeddings
        embeddings = np.stack(hiddens)
        embeddings = np.squeeze(embeddings, axis=1)

        # Performing reduction
        xy = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))

        # Add points
        for i, xy_ in zip(labels, xy):
            x, y = xy_
            plt.scatter(x, y, color=SMALL_BIG_COLORS[i])

        plt.title('Embeddings Visualization')

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markersize=10, label=language)
            for color, language in zip(SMALL_BIG_COLORS, ["Small", "Big"])
        ]
        plt.legend(handles=handles, loc='best')


class Trainer:
    """
    Class for training a language classification model.
    """

    def __init__(self, words: List[Word], model: MeaningClassifier, config: ModelConfig):
        """
        Create a new instance of Trainer.

        :param words: List of Word instances for training.
        :param model: Meaning classification model.
        :param config: Configuration for training.
        """

        self._words: List[Word] = list(words)
        self._model: MeaningClassifier = model
        self._config: ModelConfig = config

        # Training
        self._losses: List[float] = []
        self._analysis: NetworkAnalysis | None = None
        self._trained: bool = False

    def _get_training_instance(self) -> Tuple[Word, Tensor, Tensor]:
        """
        Generate a random training instance.

        :return: Tuple containing the instance word for training, the meaning tensor,
                 and the word tensor for the word.
        """
        # Choosing random language
        word = random.choice(self._words)

        # Creating tensors
        meaning_idx = Meaning.to_index(meaning=word.meaning)
        meaning_tensor = torch.tensor([meaning_idx], dtype=torch.long)
        word_tensor = self._model.alphabet.word_to_tensor(word=word.word)

        return word, meaning_tensor, word_tensor

    def train(self) -> NetworkAnalysis:
        """
        Train a language classification model using the provided configuration.

        :return: NetworkAnalysis instance for performing analysis on trained network.
        """

        if self._trained:
            return self._analysis

        # To CPU/GPU
        self._model.to(self._config.device)

        # Optimizer
        optimizer = optim.SGD(self._model.parameters(), lr=self._config.lr)

        # Criterion
        criterion = CrossEntropyLoss()

        # Loss over iterations
        self._losses = []

        for epoch in range(1, self._config.epochs + 1):

            # Get training instance
            word, meaning_tensor, word_tensor = self._get_training_instance()

            # Initial Input - Tensor of zeros
            hidden = self._model.get_starting_hidden()

            # To CPU/GPU
            meaning_tensor = meaning_tensor.to(self._config.device)
            word_tensor = word_tensor.to(self._config.device)
            hidden = hidden.to(self._config.device)

            optimizer.zero_grad()

            # Iterate over word letters
            for word_row in word_tensor:
                output, hidden = self._model(word_row, hidden)

            # Computing loss
            loss = criterion(output, meaning_tensor)

            # Backprop
            loss.backward()
            optimizer.step()

            self._losses.append(loss.item())

            # Log
            if epoch % self._config.epochs_log == 0:
                probabilities = F.softmax(output, dim=1)

                p_small = probabilities[0, 0].item()
                p_big = probabilities[0, 1].item()

                print(f"Epoch: {epoch} ({epoch * 100 / self._config.epochs}%)")
                print(f"  Loss:        {loss}")
                print(f"  Word:        {word.word}")
                print(f"  Meaning:     {word.meaning}")
                print(f"  Prob. Small: {p_small}")
                print(f"  Prob. Big:   {p_big}")
                print()

        self._trained = True
        self._analysis = NetworkAnalysis(
            model=self._model,
            losses=self._losses
        )

        return self._analysis

