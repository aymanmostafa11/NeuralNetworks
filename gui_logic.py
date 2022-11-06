import pandas as pd
import numpy as np
from DataManager import prep_data, get_viz_data
from models.NeuralNetworks import Perceptron
from Utils import accuracy_score

__model__: Perceptron
x_train: pd.DataFrame
y_train: pd.DataFrame
x_test: pd.DataFrame
y_test: pd.DataFrame
viz_data: pd.DataFrame


def fit_model(features: list, classes: list, hyper_parameters: dict):
    global __model__

    load_data(features, classes)

    __model__ = Perceptron(hyper_parameters['lr'], hyper_parameters['epochs'], hyper_parameters['bias'])
    __model__.fit(x_train, y_train)


def retrain_model():
    __model__.fit(x_train, y_train)


def test_model(train_only=False):
    """

    :return: train accuracy score and test accuracy score
    """

    train_acc = accuracy_score(y_train.values, __model__.predict(x_train))
    if train_only:
        return train_acc
    test_acc = accuracy_score(y_test.values, __model__.predict(x_test))
    return train_acc, test_acc


def get_model_weights():
    weights = np.squeeze(__model__.get_weights())
    return np.insert(weights, 0, values=0, axis=0) if len(weights) == 2 else weights # adds a bias "0" in case of no bias models


def load_data(features: list, classes: list):
    global x_train
    global y_train
    global x_test
    global y_test
    global viz_data

    x_train, x_test, y_train, y_test = prep_data(classes, features)
    viz_data = get_viz_data(classes, features)
