import numpy as np
import pandas as pd
from models import Activations
from models.Losses import mean_squared_error
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
        self._lr = lr
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
        features_count = x.shape[0]

        """
            a list of matrices where each matrix represents the weights of a particular layer,
            each matrix column represents the weights of a unit.
        """
        self._weights = []

        # Input layer "weight"
        self._weights.append(np.identity(features_count))

        # Set random weights for each layer in the network
        for i in range(1, len(self._layers)):
            self._weights.append(np.random.rand(self._layers[i]['units'], self._layers[i - 1]['units']))

        if self._bias:
            """
                a list of lists where each list represents the biases of a particular layer,
                each list element represents the bias of a unit.
            """
            self._biases = []
            # Input layer "biases"
            self._biases.append(np.zeros((self._layers[0]['units'], 1)))

            # Set random biases for each layer in the network
            for i in range(1, len(self._layers)):
                self._biases.append(np.zeros((self._layers[i]['units'], 1)))

    # Sa3ood back propagation
    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.DataFrame | np.ndarray, epochs=100, verbose=True):
        """

        Args:
            x: Input -- should be (nx,m)
            y: expected output -- should be (nL,1)
            epochs: number of iterations
        """
        # convert to numpy
        if type(x) == pd.DataFrame:
            x = x.values
        if type(y) == pd.DataFrame:
            y = y.values

        x = x.T
        y = y.T

        self.__init_weights(x)

        for i in range(0, epochs):
            AL, caches = self._forward(x)

            if verbose:
                if i % 10 == 0:
                    print(f"Epoch : {i} , Cost: {mean_squared_error(AL, y)}")

            gradients = self._backward(AL, y, caches)
            self.update_weights(gradients, self._lr)

        if verbose:
            print(f"Final Cost: {mean_squared_error(AL, y)}")
    # Forward Propagation

    def predict(self, x: pd.DataFrame | np.ndarray):
        # convert to numpy
        if type(x) == pd.DataFrame:
            x = x.values
        x = x.T
        
        return self._forward(x)[0]

    def _march(self, a_in: np.ndarray, W: np.ndarray, b: np.ndarray, g):
        """
            Applies a single forward propagation step.
            :param a_in:  input vector.
            :param W: layer weights matrix.
            :param b: layer biases list.
            :param g: layer activation function.
        """
        z = np.dot(W, a_in)
        if b is not None:
            z += b
        a_out = g(z)
        return a_out, z

    def _forward(self, x: np.ndarray):
        """
        :param x: sample input
        :return: prediction
        """

        cache = {}
        for l in range(len(self._layers)):
            bias = None if not self._bias else self._biases[l]
            x, z = self._march(x, self._weights[l], bias, self._layers[l]['activation'])
            cache["A" + str(l)] = x
            cache["Z" + str(l)] = z


        return x, cache

    # Backward Propagation

    def _retreat(self, dZ: np.ndarray, cache: dict, layer_num: int):

        """
        calculate gradiant for single layer
        Args:
            dZ: Gradient of the cost with respect to Z
            cache: dictionary where we store A , Z for computing backward

        Returns:
            dW: Gradient of the cost with respect to W
            dB: Gradient of the cost with respect to b
            dZ: Gradient of the cost with respect to Z for previous layer
        """
        A = cache["A" + str(layer_num - 1)]
        W = self._weights[layer_num]
        Z = cache["Z" + str(layer_num - 1)]
        m = A.shape[1]

        # calculate gradients for the layer
        dW = 1 / m * np.dot(dZ, A.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA = np.dot(W.T, dZ)

        if self._layers[layer_num]["activation"] == Activations.tanh:
            dZ = tanh_backward(dA, Z)

        elif self._layers[layer_num]["activation"] == Activations.sigmoid:
            dZ = sigmoid_backward(dA, Z)
        return dW, db, dZ

    def _backward(self, AL: np.ndarray, Y: np.ndarray, caches: dict):

        """
        calculate gradiant for all layers

        Args:
            AL: predictions
            Y: output
            caches: dictionary where we store A , Z for computing backward

        Returns: dictionary where we store Gradient of the cost with respect to W,b for all examples

        """

        gradients = {}
        L = len(self._layers)  # the number of layers
        m = Y.shape[1]
        dZL = AL - Y  # compute first dZ
        dZ = dZL
        for layer_num in reversed(range(1, L)):
            dW, db, dZ = self._retreat(dZ, caches, layer_num)
            gradients["dW" + str(layer_num)] = dW
            gradients["db" + str(layer_num)] = db
        return gradients

    def update_weights(self, gradients : dict, learning_rate : float):
        """
        Update parameters using gradient descent
        Arguments:
        gradients: dictionary containing your gradients
        """
        L = len(self._weights)  # number of layers in the neural network
        for layer in range(1, L):
            self._weights[layer] = self._weights[layer] - learning_rate * gradients['dW' + str(layer)]
            self._biases[layer] = self._biases[layer] - learning_rate * gradients['db' + str(layer)]

    def _calculate_cost(self):
        pass

    def _calculate_updates(self):
        pass

#
# dic = {0: {"units": 3, "activation": Activations.linear}, 1: {"units": 5, "activation": Activations.tanh},
#        2: {"units": 4, "activation": Activations.tanh}, 3: {"units": 4, "activation": Activations.tanh},
#        4: {"units": 4, "activation": Activations.tanh}, 5: {"units": 3, "activation": Activations.tanh}}
# x = [[12, 54, 65], [121, 56, 46], [561, 25, 6], [11, 5, 36], [11, 15, 61], [21, 52, 26], [1, 5, 6]]
# y = [[1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 0]]
# li1 = np.array(x).T
# li2 = np.array(y).T
#
# mod = MLP(.01, dic)
# mod.fit(li1, li2)
