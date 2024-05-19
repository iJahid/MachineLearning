import numpy as np
from ReluActivation import Relu, LeakyRelu

class FullyConnectedLayer:
    def __init__(self, name, input_size, output_size):
        self.name = name

		# Random Initialization:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size)

		# He Initialization:
        #self.weights, self.bias = self.he_initialization(input_size, output_size)

        self.relu = Relu()

        self.input = None
        self.output = None

        self.grad_weights = None
        self.grad_bias = None

    def he_initialization(self, input_size, output_size):
        '''
        He initialization for FC layer.
        size_in: the number of input units
        size_out: the number of output units
        '''
        # Compute the standard deviation
        stddev = np.sqrt(2.0 / input_size)

        # Initialize weights and bias
        weights = np.random.normal(0, stddev, (output_size, input_size))
        bias = np.zeros(output_size)

        return weights, bias

    def forward(self, mini_batch):
        self.input = mini_batch
        m = mini_batch.shape[0]
        mini_batch = mini_batch.reshape(m, -1)  # Flatten the output into a matrix of shape (m, c * h * w)
        self.output = self.relu.forward(np.dot(mini_batch, self.weights.T) + self.bias)
        return self.output

    def backprop(self, upstream_grad):
        m = self.input.shape[0]
        upstream_grad = upstream_grad.reshape(m, -1)  # Flatten the gradient into a matrix of shape (m, c * h * w)
        upstream_grad = self.relu.backprop(self.output, upstream_grad)
        grad_input = np.dot(upstream_grad, self.weights)
        self.grad_weights = np.dot(upstream_grad.T, self.input.reshape(m, -1)) / m
        self.grad_bias = np.sum(upstream_grad, axis=0) / m
        return grad_input.reshape(self.input.shape)

    def update_parameters(self, learning_rate, adam=None):
        if adam is None:
            self.weights -= learning_rate * self.grad_weights
            self.bias -= learning_rate * self.grad_bias
        else:
            parameters = {f"w{self.name}": self.weights, f"b{self.name}": self.bias}
            gradients = {f"w{self.name}": self.grad_weights, f"b{self.name}": self.grad_bias}
            adam.update(parameters, gradients, learning_rate)
            self.weights = parameters[f"w{self.name}"]
            self.bias = parameters[f"b{self.name}"]
