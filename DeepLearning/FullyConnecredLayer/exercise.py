import numpy as np
from cost_functions import CrossEntropyCost
from conv_layer import ConvolutionalLayer
from pooling_layer import PoolingLayer
from fc_layer import FullyConnectedLayer
from activation_functions import Softmax
from cost_functions import CrossEntropyCost

#####################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                               #
#                                                                   #
# TASK 1) Complete the 'FullyConnectedLayer' class in 'fc_layer.py' #
# TASK 2) Integrate your implementation in the model below          #
#####################################################################

#####################################################################
# Convolutional model                                               #
#####################################################################
np.random.seed(1)

m = 4               # Batch size
c = 3               # Depth of the input images
h = 28              # Height of the input images
w = 28              # Width of the input images
num_classes = 10    # Number of output classes
alpha = 0.01        # Learning rate

# Convolutional and Pooling layers
conv1 = ConvolutionalLayer("CONV1", input_shape=(
    c, h, w), num_filters=6, filter_size=3, stride=1, same_convolution=True)
pool1 = PoolingLayer(
    "POOL1", input_shape=conv1.output_shape, pool_size=2, stride=2)
conv2 = ConvolutionalLayer("CONV2", input_shape=pool1.output_shape,
                           num_filters=12, filter_size=3, stride=1, same_convolution=True)
pool2 = PoolingLayer(
    "POOL2", input_shape=conv2.output_shape, pool_size=2, stride=2)

# ---- YOUR TASK 2) STARTS HERE ----

# Add a fully connected layer named "FC1" with 128 output neurons
# <YOUR CODE HERE> ####
fc1 = FullyConnectedLayer(
    'FC1', input_shape=pool2.output_shape, output_neurons=128)

# Add a fully connected layer named "FC2".
# This is the last layer before the Softmax / Cross-Entropy output layer: set the output neurons accordingly.
fc2 = FullyConnectedLayer(
    'FC1', input_shape=fc1.output_shape, output_neurons=num_classes)

# ---- YOUR TASK 2) ENDS HERE ----

# Create random X and Y batches
X = np.random.randn(m, c, h, w)
Y = np.zeros((m, num_classes))
for i in range(m):
    Y[i, np.random.randint(num_classes)] = 1

# Forward Pass
A = conv1.forward(X)
A = pool1.forward(A)
A = conv2.forward(A)
A = pool2.forward(A)
A = fc1.forward(A)
A = fc2.forward(A)

# Simulate the softmax output
Y_pred = np.apply_along_axis(lambda x: np.exp(
    x) / np.sum(np.exp(x)), 1, np.random.rand(m, num_classes))

# Simulate the computation of the cost
L = -np.sum(Y * np.log(Y_pred + 1e-8)) / m

# Simulate the computation of the gradient of the cost w.r.t. the predictions: dL/dY^
dL = Y_pred - Y

# Backward Pass
dA = fc2.backprop(dL)
dA = fc1.backprop(dA)
dA = pool2.backprop(dA)
dA = conv2.backprop(dA)
dA = pool1.backprop(dA)
dX = conv1.backprop(dA)

# Parameters update
conv1.update_parameters(alpha)
conv2.update_parameters(alpha)
fc1.update_parameters(alpha)
fc2.update_parameters(alpha)
