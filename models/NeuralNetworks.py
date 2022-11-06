import numpy as np
import pandas as pd
from models.Activations import *
from Utils import accuracy_score

RANDOM_SEED = 42


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
        self.__bias = bias
        self.__random_generator__ = np.random.RandomState(RANDOM_SEED)

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame, verbose=True):

        X = X.copy(deep=True)  # to prevent adding bias from editing original data
        Y = Y.copy(deep=True)

        if self.__bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        features_count = X.shape[1]
        samples_count = X.shape[0]

        self.__weights = self.__random_generator__.rand(1, features_count)

        for epoch in range(self.__epochs):
            for i in range(samples_count):
                sample = np.array(X.iloc[i])
                activation = np.sum(sample * self.__weights)
                prediction = self.__activation(activation)
                error = Y.iloc[i] - prediction
                self.__weights += (self.__learning_rate * error * sample)

            if verbose and epoch % (self.__epochs / 10) == 0:
                acc = accuracy_score(Y, np.squeeze(self.predict(X)), verbose=False)
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

    def update_hyper_parameters(self):
        pass

    def get_weights(self):
        return self.__weights
