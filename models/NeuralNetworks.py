import numpy as np
import pandas as pd
import models.Activations
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
    def __calculate_cost(self):
        pass

    @abstractmethod
    def __calculate_updates(self):
        pass


class Adaline(Model):

    def __init__(self, lr, bias=True):
        super().__init__(lr, bias)
        self._activation = Model.Activations.linear

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, verbose=True):
        # solve using normal equation
        X = x.copy(deep=True)
        Y = y.copy(deep=True)

        if self.__bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        self._weights = np.dot(np.linalg.inv(X.T@X), X.T@Y)

    def predict(self, x: pd.DataFrame | np.ndarray):

        X = x.copy(deep=True)  # to prevent adding bias from editing original data
        if self.__bias is True and "bias" not in X.columns:
            X.insert(0, "bias", np.ones(X.shape[0]))
        return np.where(X.T@self._weights>0, 1, 0)

    def __calculate_cost(self):
        pass

    def __calculate_updates(self):
        pass


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
