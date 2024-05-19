import numpy as np

class CrossEntropyCost:
    def __call__(self, y_pred, y_true):
        epsilon = 1e-10  # to avoid log(0)
        m = y_pred.shape[0]
        cost = -np.sum(y_true * np.log(y_pred + epsilon)) / m
        return cost

    def gradient(self, y_pred, y_true):
        m = y_pred.shape[0]
        return (y_pred - y_true) / m
