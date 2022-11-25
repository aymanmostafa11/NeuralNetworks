import numpy as np

def linear(x):
    return x

def sig_num(a):
    """
        Calculate Signum activation of a weighted sum a.

        :param a: weighted sum.
        :return: 1 if a > 0, -1 otherwise.
    """
    if a == 0:
        return -1
    return int(a / abs(a))


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def tanh(a):
    return np.tanh(a)


