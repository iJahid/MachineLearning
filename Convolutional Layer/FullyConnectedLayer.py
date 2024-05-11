import numpy as np
from layer import LayerBase


class FullyConnectedLayer(LayerBase):
    def __init__(self, name, input_shape, output_neurons):
        # Initialize a Fully-Connected layer with 1024 input neurons and 'output_neurons' output neurons:
        # The weight matrix W will be of dimension (output_neurons, input_shape), transposed:
        # We work with the transposed W' = (input_shape, output_neurons) to avoid trasposition later in the dot product
        # The bias vector will be of dimension 'output_neurons'
        self.W = np.random.randn(input_shape, output_neurons)
        self.b = np.random.randn(output_neurons)
        super().__init__(name)

    def forward(self, input):
        # For this example, assume that there is no activation function
        Z = np.dot(input, self.W) + self.b
        A = Z  # We do not use any activation function to simplify the exercise
        return A

    def backprop(self, upstream_grad):
        raise NotImplementedError("Not implemented in this example")

    def update_parameters(self, learning_rate):
        raise NotImplementedError("Not implemented in this example")
