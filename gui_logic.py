import pandas as pd
import numpy as np
from DataManager import prep_data
from models.NeuralNetworks import Perceptron
import matplotlib as plt
__model__: Perceptron
X: pd.DataFrame
Y: pd.DataFrame


def fit_model(features: list, classes: list, hyper_parameters: dict):
    global __model__
    global X
    global Y

    data = prep_data(classes, features)
    __model__ = Perceptron(hyper_parameters['lr'], hyper_parameters['epochs'], hyper_parameters['bias'])
    Y = data['species']
    X = data.drop('species', axis=1)
    __model__.fit(X, Y)

    return data


def retrain_model():
    __model__.fit(X, Y)


def test_model():
    # TODO : add testing logic
    pass


def get_model_weights():
    weights = np.squeeze(__model__.get_weights())
    return np.insert(weights, 0, values=0, axis=0) if len(weights) == 2 else weights # adds a bias "0" in case of no bias models
