# The Cross-Entropy Cost (or Loss) function will be discussed in the course
# Don't worry if you do not understand it right now!

import numpy as np


class CrossEntropyCost:
    def __call__(self, y_pred, y_true):
        epsilon = 1e-10  # Valore di epsilon per evitare il log(0)
        m = y_pred.shape[0]
        cost = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return cost

    def gradient(self, y_pred, y_true):
        m = y_pred.shape[0]
        return (y_pred - y_true) / m
