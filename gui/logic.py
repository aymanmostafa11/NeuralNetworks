import pandas as pd
import numpy as np

import DataManager
from models.Losses import mean_squared_error
import models.Activations
from DataManager import prep_data, get_viz_data
from models.NeuralNetworks import Perceptron, Adaline, MLP
from models.Activations import linear, sigmoid
from Utils import accuracy_score, confusion_matrix , confusion_matrix_for_multiclass
import sklearn.metrics

__model__: Perceptron
x_train: pd.DataFrame
y_train: pd.DataFrame
x_test: pd.DataFrame
y_test: pd.DataFrame
viz_data: pd.DataFrame


def fit_model(model_name, hyper_parameters: dict, dataset: str, features: list = None, classes: list = None, is_mlp=False):
    global __model__

    # TODO: Integrate with mlp labels and different data
    load_data(dataset, features, classes, is_mlp)

    if model_name == "Perceptron":
        __model__ = Perceptron(hyper_parameters['lr'], hyper_parameters['epochs'], hyper_parameters['bias'])
        __model__.fit(x_train, y_train)

    elif model_name == "Adaline":
        __model__ = Adaline(hyper_parameters["lr"], hyper_parameters["bias"])
        __model__.fit(x_train, y_train, hyper_parameters["epochs"], hyper_parameters['min_threshold'], normal_eq=False)

    elif model_name == "MLP":
        hyper_parameters["archi"].insert(0, x_train.shape[1])
        hyper_parameters["archi"].append(y_train.shape[1])
        activation = parse_activation(hyper_parameters["activation"].lower())

        layers = {i: {"units": units, "activation": activation}
                  for i, units in enumerate(hyper_parameters["archi"])}
        layers[0]["activation"] = parse_activation("linear")

        __model__ = MLP(hyper_parameters["lr"], layers, hyper_parameters["bias"])
        __model__.fit(x_train, y_train, hyper_parameters["epochs"])



def test_model(classes=None, train_only=False, mlp=False, dataset=None):
    """
    :param dataset:
    :return: train accuracy score, test accuracy score and confusion matrix
    """
    test_pred = __model__.predict(x_test)

    if mlp:
        train_eval = mean_squared_error(y_train.values.T, __model__.predict(x_train))
        if train_only:
            return train_eval

        test_eval = mean_squared_error(y_test.values.T, test_pred)

        if dataset == "penguins":
            labels = DataManager.get_encoder("one_hot").inverse_transform([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        else:
            labels = np.array([str(i) for i in range(10)])
        conf_mat = confusion_matrix_for_multiclass(y_test.values.argmax(axis=1).tolist(),
                                                   test_pred.T.argmax(axis=1).tolist(),
                                                   np.squeeze(labels).tolist())
    else:
        train_eval = accuracy_score(y_train.values, __model__.predict(x_train))
        if train_only:
            return train_eval
        test_eval = accuracy_score(y_test.values, test_pred)
        conf_mat = confusion_matrix(y_test.values, test_pred,
                                    {"pos": classes[1], "neg": classes[0]},
                                    {"pos": 1, "neg": -1})

    return train_eval, test_eval, conf_mat


def get_model_weights():
    weights = np.squeeze(__model__.get_weights())
    return np.insert(weights, 0, values=0, axis=0) if len(weights) == 2 else weights # adds a bias "0" in case of no bias models


def load_data(dataset: str, features: list, classes: list, is_mlp=False):
    global x_train
    global y_train
    global x_test
    global y_test
    global viz_data

    x_train, x_test, y_train, y_test = prep_data(dataset, classes, features, is_mlp)
    viz_data = get_viz_data(classes, features)


def parse_activation(text):
    if text == "sigmoid":
        return models.Activations.sigmoid
    elif text == "tanh":
        return models.Activations.tanh
    elif text == "linear":
        return models.Activations.linear

    raise ValueError("Activation doesn't exist")
