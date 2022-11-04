import numpy as np
import pandas as pd
from models.Activations import *


class Perceptron:
    __activation = None
    __weights = None
    __bias = None
    __learning_rate = None
    __epochs = None

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

    def fit(self, X, Y):

        if self.__bias is True:
            X.insert(0, "bias", np.ones(X.shape[0]))

        features_count = X.shape[1]
        samples_count = X.shape[0]

        self.__weights = np.random.rand(1, features_count)

        for epoch in range(self.__epochs):
            for i in range(samples_count):
                sample = np.array(X.iloc[i])
                activation = np.sum(sample * self.__weights)
                prediction = self.__activation(activation)
                error = Y[i] - prediction
                self.__weights += (self.__learning_rate * error * sample)


    def predict(self, X):
        """
        :param X: Dataframe of samples.
        :return: numpy array of predictions.
        """
        vectorized_activation = np.vectorize(self.__activation)
        return vectorized_activation(np.dot(self.__weights, X.T))
