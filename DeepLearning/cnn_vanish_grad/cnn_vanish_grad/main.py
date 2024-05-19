import numpy as np
from TensorTools import *
from TensorMath import *
from ConvLayer import *
from AdamOptimizer import *
from FullyConnectedLayer import *
from SoftmaxActivation import *
from CrossEntropyCost import *
from dataset import *

import numpy as np

def to_categorical(y, num_classes=None):
    if num_classes is None:
        num_classes = np.max(y) + 1

    categorical = np.zeros((len(y), num_classes))
    categorical[np.arange(len(y)), y] = 1

    return categorical

def shuffle(X, Y):
    permutation = np.random.permutation(len(X))
    return X[permutation], Y[permutation]

def flatten_training_examples(X):
    return X.reshape(X.shape[0], -1)

def create_mini_batches(X_dataset, Y_dataset, m):
    num_batches = X_dataset.shape[0] // m
    return X_dataset.reshape(num_batches, m, 1, X_dataset.shape[1], X_dataset.shape[2]), Y_dataset.reshape(-1, m, Y_dataset.shape[1])

np.random.seed(184358977)

X_train, Y_train, X_test, Y_test = load_mnist_dataset("./mnist")

X_train = X_train / 255.0
Y_train = to_categorical(Y_train, 10)

input_shape = (1, 28, 28)
m = 16

# CONV 1:   input   = (m, 1, 28, 28)
#           output  = (m, 24, 28, 28)
#           24 filters 5x5
conv1_num_filters = 24
conv1_filter_size = 5
conv1 = ConvLayer("CONV1", input_shape, conv1_num_filters, conv1_filter_size, 1)

# POOL 1:	input	= (m, 24, 28, 28)
#			output	= (m, 24, 14, 14)

# CONV 2:   input   = (m, 24, 14, 14)
#           output  = (m, 48, 14, 14)
#			48 filters 5x5
conv2_num_filters = 48
conv2_filter_size = 5
conv2 = ConvLayer("CONV2", (conv1_num_filters, 14, 14), conv2_num_filters, conv2_filter_size, 1)

# POOL 2:   input   = (m, 48, 14, 14)
#           output  = (m, 48, 7, 7)

# FC 1:     input   = (m, 48 * 7 * 7)
#           output  = (m, 256)
fc1_output_neurons = 256
fc1 = FullyConnectedLayer("FC1", conv2_num_filters * 7 * 7, fc1_output_neurons)

# FC 2:     input   = (m, 256)
#           output  = (m, 10)
fc2 = FullyConnectedLayer("FC2", fc1_output_neurons, 10)

# SOFTMAX
softmax = SoftmaxActivation()

# CROSS-ENTROPY COST
cross_entropy_cost = CrossEntropyCost()

# ADAM OPTIMIZER
adam = AdamOptimizer()

epochs = 1000
learning_rate = 0.001

# Learning Rate Decay parameters
initial_learning_rate = learning_rate
decay_rate = 0.01
global_step = 0

# Epochs Loop
for epoch in range (epochs):

    # Partition the training set in mini-batches
    X_batches, Y_batches = create_mini_batches(X_train, Y_train, m)

    for i in range(X_batches.shape[0]):
        X = X_batches[i]
        Y = Y_batches[i]

        # Forward Pass:
        conv1_out = conv1.forward(X)
        pool1_out, pool1_cache = TensorMath.max_pooling(conv1_out, 2, 2)
        conv2_out = conv2.forward(pool1_out)
        pool2_out, pool2_cache = TensorMath.max_pooling(conv2_out, 2, 2)
        fc1_out = fc1.forward(pool2_out)
        fc2_out = fc2.forward(fc1_out)
        y_pred = softmax(fc2_out)
        cost = cross_entropy_cost(y_pred, Y)

        print(f"cost = {cost}")

        # Backward Pass:
        dL = cross_entropy_cost.gradient(y_pred, Y)
        upstream_grad = fc2.backprop(dL)
        assert(not np.allclose(upstream_grad, 0.0))

        upstream_grad = fc1.backprop(upstream_grad)
        assert (not np.allclose(upstream_grad, 0.0))

        upstream_grad = TensorMath.backprop_max_pooling(upstream_grad, pool2_cache)
        assert (not np.allclose(upstream_grad, 0.0))

        upstream_grad = conv2.backprop(upstream_grad)
        assert (not np.allclose(upstream_grad, 0.0))

        upstream_grad = TensorMath.backprop_max_pooling(upstream_grad, pool1_cache)
        assert (not np.allclose(upstream_grad, 0.0))

        upstream_grad = conv1.backprop(upstream_grad)
        assert (not np.allclose(upstream_grad, 0.0))

        # Exponential learning rate decay
        global_step += 1
        learning_rate = initial_learning_rate * np.exp(-decay_rate * global_step)

        # Adam optimization (parameter update)
        conv1.update_parameters(learning_rate, adam)
        conv2.update_parameters(learning_rate, adam)
        fc1.update_parameters(learning_rate, adam)
        fc2.update_parameters(learning_rate, adam)
