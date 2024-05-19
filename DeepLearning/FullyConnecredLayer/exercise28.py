import numpy as np
from conv_layer import ConvolutionalLayer
from pooling_layer import PoolingLayer
from fc_layer import FullyConnectedLayer
from output_layer import OutputLayer
from activation_functions import Softmax
from cost_functions import CrossEntropyCost
from persistence import persist, load
from training import train

#############################################################################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                                                                                       #
#                                                                                                                           #
# TASK 1) Implement the abstract methods 'persist' and 'load' in the 'ConvolutionalLayer' and 'FullyConnectedLayer' classes #
# TASK 2) Complete the new common methods 'freeze' and 'unfreeze' in 'LayerBase', and their usage in conv and fc layers     #
# TASK 3) Complete the 'persist' and 'load' functions in the 'persistence.py' file                                          #
# TASK 4) Persist the model after 'Training Phase 1' and resume it from where you left off in 'Training Phase 2'            #
# TASK 5) Load the persisted model again and continue training the fully connected layers only in 'Training Phase 3'        #
#                                                                                                                           #
# Recommendation: have a look at the new 'OutputLayer' class in 'output_layer.py'                                           #
# Recommendation: have a look at the new 'train' function in 'training.py'                                                  #
#############################################################################################################################

np.random.seed(1)

# Batch size (higher batch sizes take too long for the Udemy interactive platform)
m = 2
c = 3               # Depth of the input images
h = 28              # Height of the input images
w = 28              # Width of the input images
num_classes = 10    # Number of output classes
alpha = 0.01        # Learning rate

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
fc1 = FullyConnectedLayer("FC1", pool2.output_shape, 128)
fc2 = FullyConnectedLayer("FC2", fc1.output_shape, num_classes)
out1 = OutputLayer("OUT1", Softmax(), CrossEntropyCost())

# Insert the layers into a list
convnet = [conv1, pool1, conv2, pool2, fc1, fc2, out1]

#################################################
# Training Phase 1                              #
#################################################
# Normally we would execute the training steps within a loop, for a number of epochs.
# In this case, we perform only one training step in order to showcase the usage of pre-trained models.
L1 = train(convnet, X, Y, alpha)


# ---- YOUR TASK 4) STARTS HERE ----

# Persist the model to disk in the folder './model'
#### <YOUR CODE HERE> ####
persist(convnet, './model')

# Assume that you arrive here some days later...
# Load the persisted model from disk and continue training from where you left off
#### <YOUR CODE HERE> ####
load(convnet, './model')

#################################################
# Training Phase 2                              #
#################################################
L2 = train(convnet, X, Y, alpha)

# ---- YOUR TASK 4) ENDS HERE ----


# ---- YOUR TASK 5) STARTS HERE ----

# Reload the persisted model from disk:
# We did not persist the model after 'Training Phase 2'.
# Therefore, this operation will cancel any training progress gained with 'Training Phase 2' !
load(convnet, './model')

#################################################
# Training Phase 3                              #
#################################################
# Freeze all the convolutional layers, so that they won't be trained anymore.
# In this phase, we want to train the fully connected layers only.
# We use List Comprehension to accomplish this task.
[layer.freeze() for layer in convnet if layer.name.startswith("CONV")]

L3 = train(convnet, X, Y, alpha)


# ---- YOUR TASK 5) ENDS HERE ----
