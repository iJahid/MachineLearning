import numpy as np
from layer import LayerBase
from operations import max_pooling
# You must complete this implementation is 'operations.py'!
from operations import max_pooling_backprop


class PoolingLayer(LayerBase):

    # Constructor
    #
    # Parameters:
    # Input     name:           Name of the layer
    # Input     input_shape:    Dimensions of the input batch of feature maps X
    # Input     pool_size:      Dimension of the (square) pooling window size
    # Input     stride:         Offset used when moving the pooling window on the input feature maps
    def __init__(self, name, input_shape, pool_size, stride):

        # Initialize the internal variables
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.stride = stride

        # Compute output shape:
        # It could be convenient to first extract the input dimensions individually from 'input_shape'.
        # Remember: pooling operations only reduce the height and width of the input, while the depth (number of channels) is unchanged.
        (input_channels, input_height, input_width) = input_shape
        output_height = int((input_height - pool_size) / stride) + 1
        output_width = int((input_width - pool_size) / stride) + 1
        self.output_shape = (input_channels, output_height, output_width)

        # Call the constructor of the ancestor class
        super().__init__(name)

    # Forward Pass
    #
    # Parameters:
    # Input     X:  Batch of input feature maps
    # Output        Batch containing the downsampled input feature maps. The number of channels is unchanged.
    def forward(self, X):

        # Call the 'max_pooling' function that we have already implemented
        # Note: the result is a dictionary containing the output and the cache
        result = max_pooling(X, self.pool_size, self.stride)

        # Extract the operation cache containing (X, pool_size, stride) and store it to the internal variable 'cache'
        self.cache = result["cache"]

        # Return the batch of downsampled feature maps
        return result["output"]

    # Backward Pass
    #
    # Parameters:
    # Input     upstream_grad:  Batch containing the gradients of the cost w.r.t the outputs of the pooling layer (dL/dA)
    # Output    Batch containing the gradients of the cost w.r.t the input feature maps (dL/dX)
    def backprop(self, upstream_grad):

        # Use the 'max_pooling_backprop' procedure that you have defined in 'operations.py'
        return max_pooling_backprop(upstream_grad, self.cache)
