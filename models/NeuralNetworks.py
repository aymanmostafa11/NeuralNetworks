import numpy as np
import pandas as pd
from models import Activations
import models.Losses
from models.Activations import *
from Utils import accuracy_score
from abc import ABC, abstractmethod

RANDOM_SEED = 42
RANDOM_GENERATOR = np.random.RandomState(RANDOM_SEED)


class Model(ABC):
    def __init__(self, lr, bias=True):
        self._lr = lr
        self._bias = bias
        self._activation = None
        self._weights = None

    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, verbose=True):
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray):
        pass

    def __init_weights(self, x: pd.DataFrame):
        """
        Initializes random weights with shape (1, features)
        """
        features_count = x.shape[1]
        if self._bias:
            features_count += 1
        self.__weights = RANDOM_GENERATOR.rand(1, features_count)

    @abstractmethod
    def _calculate_cost(self):
        pass

    @abstractmethod
    def _calculate_updates(self):
        pass

    def get_weights(self):
        return self._weights


class Adaline(Model):

    def __init__(self, lr, bias=True):
        super().__init__(lr, bias)
        self._activation = Activations.linear

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, verbose=True):
        # solve using normal equation
        X = x.copy(deep=True)
        Y = y.copy(deep=True)

        if self._bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        self._weights = np.dot(np.linalg.inv(X.T @ X), X.T @ Y)

    def predict(self, x: pd.DataFrame | np.ndarray):

        X = x.copy(deep=True)  # to prevent adding bias from editing original data
        if self._bias is True and "bias" not in X.columns:
            X.insert(0, "bias", np.ones(X.shape[0]))
        return np.where(X @ self._weights > 0, 1, -1)

    def _calculate_cost(self):
        print("Not Implemented")

    def _calculate_updates(self):
        print("Not Implemented")


class Perceptron:

    def __init__(self, learning_rate, epochs, bias=True, activation=sig_num):
        """
        :param activation: The activation function to be used, default is sig_num().
        :param learning_rate: The Learning rate.
        :param epochs: Number of epochs.
        :param bias: Boolean to determine wether to add a bias or not, default is True.
        """
        self.__activation = activation
        self.__learning_rate = learning_rate
        self.__epochs = epochs
        self.__weights = None
        self.__bias = bias

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, verbose=True):
        # TODO : change epochs to be included in here instead of init
        X = X.copy(deep=True)  # to prevent adding bias from editing original data
        Y = Y.copy(deep=True)

        if self.__bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        features_count = X.shape[1]
        samples_count = X.shape[0]

        self.__weights = RANDOM_GENERATOR.rand(1, features_count)

        for epoch in range(self.__epochs):
            for i in range(samples_count):
                sample = np.array(X.iloc[i])
                activation = np.sum(sample * self.__weights)
                prediction = self.__activation(activation)
                error = Y.iloc[i] - prediction
                self.__weights += (self.__learning_rate * error * sample)

            if verbose and epoch % (self.__epochs / 10) == 0:
                acc = accuracy_score(Y.values, np.squeeze(self.predict(X)), verbose=False)
                print(f"Epoch : {epoch}, accuracy : {acc}")

    def predict(self, X):
        """
        :param X: Dataframe of samples.
        :return: numpy array of predictions.
        """
        X = X.copy(deep=True)  # to prevent adding bias from editing original data
        if self.__bias is True and "bias" not in X.columns:
            X.insert(0, "bias", np.ones(X.shape[0]))

        vectorized_activation = np.vectorize(self.__activation)
        return np.squeeze(vectorized_activation(np.dot(self.__weights, X.T)))

    def get_weights(self):
        return self.__weights


class MLP(Model):
    """
        :param lr: learning rate
        :param layers: a list of dicts holding layer info, key = layer number, input is layers[0]
        :param bias: bias
    """

    def __init__(self, lr, layers: dict, bias=True):
        super().__init__(lr, bias)
        self._layers = layers
        """
            layers
            [
                INPUT LAYER
                { units: inputs_size, activation: linear },
                HIDDEN LAYERS
                { units: n_units, activation: RELU },
                { units: n_units, activation: RELU },
                OUTPUT LAYER
                { units: 1, activation: sigmoid },
            ]
            IMPORTANT: make sure to give the input layer a linear activation
        """

    def __init_weights(self, x: pd.DataFrame):
        features_count = x.shape[1]

        """
            a list of matrices where each matrix represents the weights of a particular layer,
            each matrix column represents the weights of a unit.
        """
        self._weights = []

        # Input layer "weight"
        self._weights.append(np.identity(features_count))

        # Set random weights for each layer in the network
        for i in range(1, len(self._layers)):
            self._weights.append(np.random.rand(self._layers[i - 1]['units'], self._layers[i]['units']))

        if self._bias:
            """
                a list of lists where each list represents the biases of a particular layer,
                each list element represents the bias of a unit.
            """
            self._biases = []
            # Input layer "biases"
            self._biases.append(np.zeros(self._layers[0]['units']))

            # Set random biases for each layer in the network
            for i in range(1, len(self._layers)):
                self._biases.append(np.random.rand(self._layers[i]['units']))

    # Sa3ood back propagation
    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, verbose=True):
        pass

    # Forward Propagation
    def predict(self, x: pd.DataFrame | np.ndarray):
        return self._forward(x)

    def _march(self, a_in: np.ndarray, W: np.ndarray, b: np.ndarray, g):
        """
            Applies a single forward propagation step.
            :param a_in:  input vector.
            :param W: layer weights matrix.
            :param b: layer biases list.
            :param g: layer activation function.
        """
        z = a_in @ W + b
        a_out = g(z)
        return a_out

    def _forward(self, x: np.ndarray):
        """
        :param x: sample input
        :return: prediction
        """
        for i in range(len(self._layers)):
            x = self._march(x, self._weights[i], self._biases[i], self._layers[i]['activation'])
        return x
