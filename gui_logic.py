import pandas as pd
from DataManager import prep_data
from models.NeuralNetworks import Perceptron
import matplotlib as plt
__model__ : Perceptron
X : pd.DataFrame
Y : pd.DataFrame

def fit_model(features: list, classes: list, hyper_parameters: dict):
    data = prep_data(classes,features)
    __model__ = Perceptron(hyper_parameters['lr'],hyper_parameters['epochs'])
    Y = data['species']
    X = data.drop('species', axis=1)
    __model__.fit(X, Y)
    return data


def retrain_model():
    __model__.fit(X, Y)


def test_model():
    # TODO : add testing logic
    pass

