import numpy as np
import abc
np.random.seed(1)

# Base class for all layers in our convolutional neural network
# Complete the following code:


class LayerBase:

    # Define the constructor taking the layer name as a parameter
    # Input Parameters:     'name' string with the layer name
    # Actions:              set class member
    def __init__(self, name):
        self.name = name
    #### <YOUR CODE HERE> ####

    # Define the abstract method 'forward':
    # Input Parameters:     'input' tensor
    # Output Parameters:    none (the abstract method does not return any values)
    # Actions:              raise an exception inviting to implement the method in the subclass

    #### <YOUR CODE HERE> ####
    @abc.abstractmethod
    def forward(self, input):
        raise NotImplementedError

    # Define the abstract method 'backprop':
    # Input Parameters:     'upstream_grad' tensor
    # Output Parameters:    none (the abstract method does not return any values)
    # Actions:              raise an exception inviting to implement the method in the subclass
    @abc.abstractmethod
    def backprop(self, upstream_grad):
        raise NotImplementedError
    #### <YOUR CODE HERE> ####

    # Define the abstract method 'update_parameters':
    # Input Parameters:     'learning_rate' value
    # Output Parameters:    none (the abstract method does not return any values)
    # Actions:              use the 'pass' keyword to give the possibility of overriding the method in subclasses
    def update_parameters(self, learning_rate):
        self.learning_rate = learning_rate
        pass
    #### <YOUR CODE HERE> ####

#####################################################################################
# The following example convolutional layer inherits from LayerBase                 #
# This conv layer is not functional: its only purpose is to validate inheritance!   #
# Do not modify the following code: it is used as a test-case!                      #
# But you are encouraged to take a look at it!                                      #
#####################################################################################


class ConvLayerExample(LayerBase):
    def __init__(self, name, input_shape, num_filters, filter_size, stride):
        self.input_shape = input_shape  # (in_channels, in_height, in_width)
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride

        (in_channels, in_height, in_width) = input_shape
        self.filters = np.random.randn(
            num_filters, in_channels, filter_size, filter_size)
        self.bias = np.zeros(num_filters)

        # IMPORTANT: Here we call the ancestor class's constructor:
        # The 'name' parameter is inherited from the ancestor class
        super().__init__(name)

    # Concrete implementation of the abstract method 'forward'
    def forward(self, input):
        # mock action showing that forward is being called
        print(f"{self.name} input: {input}")

    # Concrete implementation of the abstract method 'backprop'
    def backprop(self, upstream_grad):
        # mock action showing that backprop is being called
        print(f"{self.name} gradient: {upstream_grad}")

    # Concrete implementation of the abstract method 'update_parameters'
    def update_parameters(self, learning_rate):
        # mock action showing that update_parameters is being called
        print(f"updating {self.name} with lr={learning_rate}")


# Test code
x_shape = (3, 4, 4)  # 3x4x4 image
X = np.random.randn(*x_shape)  # fake input
dA = np.random.randn(*x_shape)  # fake upstream gradients
alpha = 0.002

layer = ConvLayerExample("CONV1", input_shape=x_shape,
                         num_filters=8, filter_size=3, stride=1)

layer.forward(X)
layer.backprop(dA)
layer.update_parameters(alpha)
