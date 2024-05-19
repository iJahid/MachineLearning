from persistence import write_parameters, read_parameters
import numpy as np
from conv_layer import ConvolutionalLayer
from fc_layer import FullyConnectedLayer

#####################################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                                               #
#                                                                                   #
# TASK 1) Complete the 'write_parameters' function in the 'persistence.py' file     #
# TASK 2) Complete the 'read_parameters' function in the 'persistence.py' file      #
# TASK 3) Complete the 'get/set_parameters' functions in the ConvolutionalLayer     #
# TASK 4) Complete the 'get/set_parameters' functions in the FullyConnectedLayer    #
# TASK 5) Use the new functions to store and load the parameters to and from disk   #
#                                                                                   #
# Recommendation: have a look at get/set_parameters in the LayerBase class:         #
# Note that the get/set_parameters functions are abstract in the base class.        #
#####################################################################################

np.random.seed(1)

c = 3               # Depth of the input images
h = 28              # Height of the input images
w = 28              # Width of the input images
num_classes = 10    # Number of output classes

conv1 = ConvolutionalLayer("CONV1", input_shape=(
    c, h, w), num_filters=6, filter_size=3, stride=1, same_convolution=True)
conv2 = ConvolutionalLayer("CONV2", input_shape=conv1.output_shape,
                           num_filters=12, filter_size=3, stride=1, same_convolution=True)
fc1 = FullyConnectedLayer("FC1", conv2.output_shape, 128)
fc2 = FullyConnectedLayer("FC2", fc1.output_shape, num_classes)

# ---- YOUR TASK 5) STARTS HERE ----

# Import the persistence functions


# Retrieve the parameters of the layers
conv1_params = conv1.get_parameters()
conv2_params = conv2.get_parameters()
fc1_params = fc1.get_parameters()
fc2_params = fc2.get_parameters()

# Persist the parameters to binary files located in the './model' folder
write_parameters(conv1_params, 'model')
write_parameters(conv2_params, 'model')
write_parameters(fc1_params, 'model')
write_parameters(fc2_params, 'model')

# Load the parameters from the binary files located in the './model' folder
# Note: we load the parameters into different variables to allow the test-case to compare them with the original parameters
loaded_conv1_params = read_parameters('model', 'CONV1')
loaded_conv2_params = read_parameters('model', 'CONV2')
loaded_fc1_params = read_parameters('model', 'fc1')
loaded_fc2_params = read_parameters('model', 'fc2')

# Set the parameters to the layers
conv1.set
#### <YOUR CODE HERE> ####
#### <YOUR CODE HERE> ####
#### <YOUR CODE HERE> ####

# ---- YOUR TASK 5) ENDS HERE ----
