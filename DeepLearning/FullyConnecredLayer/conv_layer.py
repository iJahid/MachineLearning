import numpy as np
from operations import convolution, convolution_backprop
from activation_functions import Relu
from layer import LayerBase
from persistence import read_parameters, write_parameters


class ConvolutionalLayer(LayerBase):

    # Constructor:
    # The same conv layer instance should be compatible with input batches having a different number of examples.
    # Therefore, the input dimensions specified in the constructor do not consider the batch size 'm'.
    # The parameter 'same_convolution' is optional, specifying whether this layer should perform 'same convolutions' or not.
    # If not specified, the convolutional layer will perform 'same convolutions', preserving the height and width of feature maps.
    def __init__(self, name, input_shape, num_filters, filter_size, stride, same_convolution=True):
        # (in_channels, in_height, in_width). Note: the shape does not contain the batch size m.
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride

        # Determine the padding to apply, depending on whether "same convolutions" are requested
        self.padding = (filter_size - 1) // 2 if same_convolution else 0

        # Initialize filters randomly in the range [0, 1] and biases to 0
        (in_channels, in_height, in_width) = input_shape
        self.filters = np.random.randn(
            num_filters, in_channels, filter_size, filter_size)
        self.biases = np.zeros(num_filters)

        # Create the Relu activation function
        self.activation = Relu()

        # Determine the output shape:
        # When performing a same convolution, the features' height and width remain unchanged.
        # As a result, the output volume will have the same height and width as the input volume.
        # When same convolution is not requested, the output height and width will shrink.
        # You have to calculate the output height and width using a formula that takes padding into consideration.
        output_height = (
            self.input_shape[1] + 2 * self.padding - self.filter_size) // self.stride + 1
        output_width = (
            self.input_shape[2] + 2 * self.padding - self.filter_size) // self.stride + 1

        # Define the internal variable 'output_shape' using the calculated output height and width.
        # Remember: the depth (number of channels) of the output volume is the number of filters!
        # Remember: the output shape does not consider the batch size!
        self.output_shape = (num_filters, output_height, output_width)

        # Call the constructor of the LayerBase class
        super().__init__(name)

    # Retrieve the layer's parameters
    #
    # Parameters:
    # Output    Dictionary containing the layer's name and parameters.
    def get_parameters(self):
        return {
            "layer_name": self.name,
            "filters": self.filters,
            "biases": self.biases
        }

    # Set the layer's parameters
    #
    # Parameters:
    # Input     parameters:     Dictionary containing the layer's parameters.
    def set_parameters(self, parameters):
        self.filters = parameters["filters"]
        self.biases = parameters["biases"]

    # ---- YOUR TASK 1) STARTS HERE ----

    # Persist the parameters of this layer to disk
    #
    # Parameters:
    # Input     folder:     Destination folder into which the parameters will be stored.
    def persist(self, folder):

        # Use the 'get_parameters' function to retrieve a ready-to-use dictionary containing the parameters
        parameters = self.get_parameters()

        # Store the parameters using the persistence functions which you implemented in the previous exercise
        write_parameters(parameters, folder)

    # Load the parameters of this layer from disk
    #
    # Parameters:
    # Input     folder:     Source folder from which the parameters will be loaded.
    def load(self, folder):

        # Load the parameters using the persistence functions which you implemented in the previous exercise
        parameters = read_parameters(folder, self.name)

        # Use the 'set_parameters' function to set the loaded parameters
        self.set_parameters(parameters)

    # ---- YOUR TASK 1) ENDS HERE ----

    # Forward Pass:
    # This procedure uses the 'convolution' operation that we have implemented. You can find it in the 'operations.py' file.
    #
    # Parameters:
    # Input     X:  Batch of input examples
    # Output    A:  The activation of the layer corresponding to the input batch X
    def forward(self, X):

        # Use the function that you have implemented in Section 5, Coding Exercise 8: "Implement the Convolution Operation".
        # The correct implementation of the 'convolution' function is available in the 'operations.py' file.
        result = convolution(X, self.filters, self.biases,
                             self.stride, self.padding)

        # Store the convolution cache to the internal variable 'cache'
        self.cache = result["cache"]

        # Extract the convolution result: this is the intermediate output of the layer, before the activation.
        # This is what we normally call 'Z'.
        Z = result["output"]

        # Compute the activation of 'Z', store it to the internal variable 'prev_activation', and return it.
        # The internal variable 'prev_activation' contains, at any time, the last activation of the layer.
        # The last activation is used later to back-propagate the upstream gradient through the activation function.
        A = self.activation.forward(Z)
        self.prev_activation = A
        return A

    # Backward Pass (Backprop)
    # This procedure computes the gradients of the cost w.r.t. the filters (dL/dW), the biases (dL/db), and the input batch (dL/dX).
    # The gradients of the cost w.r.t. the input batch, dL/dX, is returned as an output to continue the Backpropagation.
    # In the future, you could consider moving the internal backprop operation to 'operations.py' and call it from this procedure.
    # This way, you would use a consistent approach in the Forward and Backward Pass.
    #
    # Parameters:
    # Input     upstream_grad:  Upstream batch containing the gradients of the cost w.r.t. the activation of this layer: dL/dA
    # Output    The gradients of the cost w.r.t. the input batch X: dL/dX -- this can be seen as dL/dA for the previous layer.
    def backprop(self, upstream_grad):

        # First of all, we have to back-propagate the upstream gradient through the activation unit.
        # Remember: 'upstream_gradient' is the gradient of the cost w.r.t. the previous activation of this layer: dL/dA (or just dA).
        # In order to compute the gradient of the cost w.r.t. the filters (dL/dW), the biases (dL/db) and the input (dL/dX),
        # we have to compute dL/dZ first (or just dZ).
        # Since Z is the intermediate output of the layer before the activation, we can obtain dZ by back-propagating dA through
        # the activation unit. In general, dZ = activation.backprop(dA).
        # Once we have obtained dZ, we can compute dW, db, and dX as functions of dZ.
        #
        # Note: we have back-propagated the upstream gradient (dA) through the activation unit and we obtained dZ:
        # We now consider dZ our new "upstream" gradient. From now on, we refer to dZ when talking about the "upstream" gradient.
        dZ = self.activation.backprop(self.prev_activation, upstream_grad)

        # From the cache, extract the padded input of the previous forward pass.
        # Hint: the cache is a tuple, so the items are stored in it positionally. Check the chache initialization in 'operations.py' to find out at which position the padded input X_pad was stored!
        X_pad = self.cache["X_pad"]

        # Compute the gradients dL/dW, dL/db, dL/dX
        self.grad_filters, self.grad_biases, grad_input = convolution_backprop(
            self.input_shape, self.filters, self.stride, self.biases, dZ, X_pad, self.padding, upstream_grad
        )

        return grad_input[:, :, self.padding:self.input_shape[1] + self.padding, self.padding:self.input_shape[2] + self.padding]

    # Update Parameters:
    # This procedure is used to update the filters and the biases based on their gradients: dL/dW and dL/db, respectively.
    #
    # Parameters:
    # Input     learning_rate:  Learning rate used for Gradient Descent
    def update_parameters(self, learning_rate):

        # ---- YOUR TASK 2) STARTS HERE ----

        if not self.frozen:
            # Update the filters and the biases according to the Gradient Descent formulas:
            self.filters -= learning_rate * self.grad_filters
            self.biases -= learning_rate * self.grad_biases

        # ---- YOUR TASK 2) STARTS HERE ----
