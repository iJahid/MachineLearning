import numpy as np
from conv_layer import ConvolutionalLayer
from pooling_layer import PoolingLayer
from activation_functions import Softmax
from cost_functions import CrossEntropyCost

#############################################################################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                                                                                       #
#                                                                                                                           #
# TASK 1) Define the first fully connected layer "FC1" as a convolutional layer                                             #
# TASK 2) Define the second fully connected layer "FC2" as a convolutional layer                                            #
# TASK 3) Manipulate the dimensions of FC2's output in the Forward Pass to make them compatible with the softmax            #
# TASK 4) Manipulate the dimensions of the upstream gradient to FC2 to make it compatible with the layer's output shape     #
#############################################################################################################################

np.random.seed(1)

m = 8               # Batch size
c = 3               # Depth of the input images
h = 28              # Height of the input images
w = 28              # Width of the input images
num_classes = 10    # Number of output classes

# Create random X and Y batches
X = np.random.randn(m, c, h, w)
Y = np.zeros((m, num_classes))
for i in range(m):
    Y[i, np.random.randint(num_classes)] = 1

# Define the layers of the convolutional model
conv1 = ConvolutionalLayer("CONV1", input_shape=(
    c, h, w), num_filters=4, filter_size=3, stride=1, same_convolution=True)
pool1 = PoolingLayer(
    "POOL1", input_shape=conv1.output_shape, pool_size=2, stride=2)
conv2 = ConvolutionalLayer("CONV2", input_shape=pool1.output_shape,
                           num_filters=8, filter_size=3, stride=1, same_convolution=True)
pool2 = PoolingLayer(
    "POOL2", input_shape=conv2.output_shape, pool_size=2, stride=2)

# ---- YOUR TASK 1) STARTS HERE ----

# The (convolutional) fully-connected layer FC1 has 128 output neurons
# The output of FC1 will be a batch of feature maps of shape (128, 1, 1)
fc1 = ConvolutionalLayer(
    "FC1",
    input_shape=pool2.output_shape,
    num_filters=128,
    filter_size=pool2.output_shape[1],
    stride=1,
    same_convolution=False  # we must not preserve height and width of the input feature maps!
)

# ---- YOUR TASK 1) ENDS HERE ----

# ---- YOUR TASK 2) STARTS HERE ----

# The (convolutional) fully-connected layer FC2 must have 'num_classes' output neurons
# The input of FC2 is a batch of feature maps of shape (128, 1, 1) -- outputted by FC1
fc2 = ConvolutionalLayer(
    "FC2",
    input_shape=fc1.output_shape,
    num_filters=num_classes,
    filter_size=1,
    stride=1,
    same_convolution=False  # we must not preserve height and width of the input feature maps!
)

# ---- YOUR TASK 2) ENDS HERE ----

softmax = Softmax()
cost_fn = CrossEntropyCost()

# Forward Pass
A = conv1.forward(X)
A = pool1.forward(A)
A = conv2.forward(A)
A = pool2.forward(A)
A = fc1.forward(A)
A = fc2.forward(A)

# ---- YOUR TASK 3) STARTS HERE ----

# When using a convolutional implementation of the FC layer, the output is of shape: (m, num_filters, 1, 1).
# The softmax activation requires a batch of 1-dimensional vectors of shape: (m, output_classes).
# We know that the last (convolutional) FC layer produces an output of shape: (m, output_classes, 1, 1).
# Therefore, we must squeeze FC2's output into a volume of shape (m, output_classes): this shape is compatible with the softmax unit.
# In order to accomplish this, we can use the numpy 'squeeze' function, which removes all dimensions with length 1.

A = np.squeeze(A)  # <YOUR CODE HERE> ####

# ---- YOUR TASK 3) ENDS HERE ----

# Compute the predictions: Y^
Y_pred = softmax(A)

# Compute the cost: L
L = cost_fn(Y_pred, Y)

# Compute the gradient of the cost w.r.t. the predictions: dL/dY^
dL = cost_fn.gradient(Y_pred, Y)

# ---- YOUR TASK 4) STARTS HERE ----

# When using a convolutional implementation of the FC layer, we have to consider the following:
# The output of the last (convolutional) FC layer before the softmax activation was of shape: (m, output_classes, 1, 1).
# The gradient of the cost w.r.t. the predictions (dL/dY^) is of shape: (m, output_classes).
# In order for dL/dY^ to be compatible with the last (convolutional) FC layer, we must resize dL/dY^ to (m, output_classes, 1, 1).
# We can accomplish this by expanding it with two additional dimensions, using the numpy 'expand_dims' function.
dL = np.expand_dims(dL, axis=2)
dL = np.expand_dims(dL, axis=3)

# ---- YOUR TASK 4) ENDS HERE ----

# Backward Pass
dA = fc2.backprop(dL)
dA = fc1.backprop(dA)
dA = pool2.backprop(dA)
dA = conv2.backprop(dA)
dA = pool1.backprop(dA)
dX = conv1.backprop(dA)
