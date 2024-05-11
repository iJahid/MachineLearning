import numpy as np
from cost_functions import CrossEntropyCost
from conv_layer import ConvolutionalLayer
from pooling_layer import PoolingLayer
from fc_layer import FullyConnectedLayer
from activation_functions import Softmax
#####################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                               #
#                                                                   #
# TASK 1) Import the Softmax and CrossEntropyCost modules           #
# TASK 2) Integrate the Softmax / Cross-Entropy output layer	    #
# TASK 3) Use the Softmax / Cross-Entropy output layer		        #
#####################################################################

# ---- YOUR TASK 1) STARTS HERE ----


# ---- YOUR TASK 1) ENDS HERE ----

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
fc1 = FullyConnectedLayer("FC1", pool2.output_shape, 128)
fc2 = FullyConnectedLayer("FC2", fc1.output_shape, num_classes)

# ---- YOUR TASK 2) STARTS HERE ----

# Integrate the Softmax activation unit
softmax = Softmax()  # <YOUR CODE HERE> ####

# Integrate the Cross-Entropy cost function
# Note: many researchers and authors use the term "loss" instead of "cost".
cost_fn = CrossEntropyCost()  # <YOUR CODE HERE> ####

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

# ---- YOUR TASK 3) STARTS HERE ----


# Compute the predictions: Y^
Y_pred = softmax(A)

# Compute the cost: L
L = cost_fn(Y_pred, Y)

# Compute the gradient of the cost w.r.t. the predictions: dL/dY^
dL = cost_fn.gradient(Y_pred, Y)

# ---- YOUR TASK 3) ENDS HERE ----

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
