import numpy as np


def to_one_hot(labels, num_classes):
    one_hot = np.zeros([len(labels), num_classes])
    one_hot[range(len(one_hot)), labels] = 1
    return one_hot


def from_one_hot(one_hot):
    return np.argmax(one_hot)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_output_to_derivative(x):
    return x * (1 - x)


def ReLU(x):
    x[ x < 0 ] = 0
    return x


def ReLU_derivative(x):
    x[ x > 0 ] = 1
    return x


def accuracy(predictions, true):
    return np.mean(true == predictions)
