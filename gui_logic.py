import pandas as pd

from DataManager import prep_data
from models.NeuralNetworks import Perceptron

__model__ : Perceptron
X : pd.DataFrame
Y : pd.DataFrame

def fit_model(features: list, classes: list, hyper_parameters: dict):
    data = prep_data()
    __model__ = Perceptron(learning_rate=0.1, epochs=10)
    Y = data['body_mass_g']
    X = data.drop('body_mass_g', axis=1)
    __model__.fit(X, Y)


def retrain_model():
    __model__.fit(X, Y)


def test_model():
    # TODO : add testing logic
    pass