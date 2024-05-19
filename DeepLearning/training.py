import numpy as np
# Simulate one training step
#
# Parameters:
# Input     X:  Batch of training examples
# Input     Y:  Batch of labels


def train(X, Y, learning_rate):
    conv1 = np.random.randn(16, 6, 28, 28)
    pool1 = np.random.randn(16, 6, 14, 14)
    conv2 = np.random.randn(16, 8, 14, 14)
    pool2 = np.random.randn(16, 8, 7, 7)
    fc1 = np.random.randn(128)
    fc1 = np.random.randn(10)
