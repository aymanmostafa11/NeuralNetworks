import numpy as np


def mean_squared_error(pred, actual):
    return np.round(np.sum(np.power((pred - actual), 2)) / (2 * len(actual)), 4)
