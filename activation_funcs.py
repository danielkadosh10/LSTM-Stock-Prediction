# This file contains all the activation functions we plan to support.
import numpy as np


class func_type(enumerate):
    SIGMOID = 1
    TANH = 2


# y = 1 / (1 + e^(-x))
def sigmoid(x):
    y = 1 / (1 + np.exp(-1 * x))

    return y


# y = [e^(2x) - 1] / [e^(2x) + 1]
def tanh(x):
    y = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    return y
