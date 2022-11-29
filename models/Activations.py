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


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_backward(dA, Z):
    """
     Arguments:
     dA : previous-activation gradient
     Z : for computing backward propagation

     Returns:
     dZ -- Gradient of the cost with respect to Z
     """
    
    a = sigmoid(Z)
    dZ = dA * a * (1-a)

    assert (dZ.shape == Z.shape)

    return dZ

def tanh(Z):
    return np.tanh(Z)

def tanh_backward(dA, Z):
    """
    Arguments:
    dA : previous-activation gradient
    Z : for computing backward propagation

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    a = tanh(Z)
    dZ = dA * (1- np.power(a,2))

    assert (dZ.shape == Z.shape)

    return dZ

