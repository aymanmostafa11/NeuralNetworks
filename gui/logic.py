import pandas as pd
import numpy as np
from DataManager import prep_data, get_viz_data
from models.NeuralNetworks import Perceptron, Adaline
from Utils import accuracy_score, confusion_matrix

__model__: Perceptron
x_train: pd.DataFrame
y_train: pd.DataFrame
x_test: pd.DataFrame
y_test: pd.DataFrame
viz_data: pd.DataFrame


def fit_model(model_name, features: list, classes: list, hyper_parameters: dict):
    global __model__

    load_data(features, classes)

    if model_name == "Perceptron":
        __model__ = Perceptron(hyper_parameters['lr'], hyper_parameters['epochs'], hyper_parameters['bias'])
        __model__.fit(x_train, y_train)

    elif model_name == "Adaline":
        __model__ = Adaline(hyper_parameters["lr"], hyper_parameters["bias"])
        __model__.fit(x_train, y_train, hyper_parameters["epochs"])


def test_model(classes=None, train_only=False):
    """
    :return: train accuracy score, test accuracy score and confusion matrix
    """

    train_acc = accuracy_score(y_train.values, __model__.predict(x_train))
    if train_only:
        return train_acc

    test_pred = __model__.predict(x_test)
    test_acc = accuracy_score(y_test.values, test_pred)

    conf_mat = confusion_matrix(y_test.values, test_pred,
                                {"pos": classes[1], "neg": classes[0]},
                                {"pos": 1, "neg": -1})
    return train_acc, test_acc, conf_mat


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
