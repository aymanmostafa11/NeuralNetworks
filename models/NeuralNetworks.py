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

    def _init_weights(self, x: pd.DataFrame):
        """
        Initializes random weights with shape (1, features)
        """
        features_count = x.shape[1]
        if self._bias:
            features_count += 1
        self._weights = RANDOM_GENERATOR.rand(1, features_count)

    @abstractmethod
    def _calculate_cost(self, pred, actual):
        pass

    def get_weights(self):
        return self._weights




class Adaline(Model):

    def __init__(self, lr, bias=True):
        super().__init__(lr, bias)
        self._activation = Activations.linear

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, min_threshold=1.0, normal_eq=True, verbose=True):
        """
        :param x: Training vector
        :param y: Target vector relative to x
        :param epochs: Specifies number of times to loop over whole Data, default=100, discarded if normal_eq=True
        :param min_threshold: Specifies minimum cost to stop fitting the model, default=1.0, discarded if normal_eq=True
        :param normal_eq: Solves for weights analytically using least squares cost function, default:False
        :param verbose: Prints cost after each epoch, default=True, , discarded if normal_eq=True
        """
        X = x.copy(deep=True)
        Y = y.copy(deep=True)

        self._init_weights(X)

        if self._bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        if normal_eq == True:
            self._solve_analytically(X, Y)
        else:
             self._solve_iteratively(X, Y, epochs, min_threshold, verbose)

    def _solve_iteratively(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs, min_threshold, verbose):

        samples_count = X.shape[0]

        for e in range(epochs):
            for i in range(samples_count):

                sample = np.array(X.iloc[i])
                output = np.dot(self._weights, sample)
                error = y.iloc[i]-output
                self._weights = np.add(self._weights, self._lr*error*sample.T)

            # calculate error over all samples using updated weights
            cost = self._calculate_cost(self.predict(X), y)
            if verbose==True:
                print(f'epoch {e}: {cost}')
            if cost <= min_threshold:
                break

    def _solve_analytically(self, X: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray):
        self._weights = np.dot(np.linalg.inv(X.T @ X), X.T @ y)

    def predict(self, x: pd.DataFrame | np.ndarray):

        X = x.copy(deep=True)
        if self._bias is True and "bias" not in X.columns:
            X.insert(0, "bias", np.ones(X.shape[0]))
        return np.where(self._weights@X.T > 0, 1, -1)

    def _calculate_cost(self, pred, actual):
        return np.sum(np.power((pred-np.array(actual)), 2))/(2*len(actual))




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
