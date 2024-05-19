import numpy as np
from TensorMath import *
from ReluActivation import Relu, LeakyRelu

class ConvLayer:
    def __init__(self, name, input_shape, num_filters, filter_size, stride, same_convolution=True):
        self.name = name
        self.input_shape = input_shape
        (input_channels, input_height, input_width) = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride

        # Random Initialization:
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size)

        # He Initialization:
        #self.filters = self.he_initialization_filters(num_filters, input_channels, filter_size)

        self.bias = np.zeros((num_filters, 1, 1, 1))

        self.relu = Relu()
        self.same_convolution = same_convolution

        self.grad_filters = None
        self.grad_bias = None

        self.output = None
        self.cache = None

        if same_convolution:
            self.padding = int((self.filter_size - 1) / 2)
        else:
            self.padding = 0

        # Determine the output shape:
        # If we are performing a same convolution, the output volume will maintain height and width, whereas its depth will be the number of filters
        output_height = int((input_height - filter_size + 2 * self.padding) / stride) + 1
        output_width = int((input_width - filter_size + 2 * self.padding) / stride) + 1
        self.output_shape = (num_filters, output_height, output_width)

    def he_initialization_filters(self, num_filters, input_channels, filter_size):
        """
        He initialization of filters for a conv layer.

        Parameters:
        num_filters: int, number of filters
        input_channels: int, number of input channels
        filter_size: int, size of the filter

        Returns:
        filters: np.ndarray, initialized filters
        """

        # Calcola la varianza come suggerito da He et al.
        stddev = np.sqrt(2. / (num_filters * input_channels * filter_size * filter_size))

        # Crea i filtri con valori estratti da una normale con media 0 e deviazione standard calcolata
        filters = np.random.normal(0, stddev, size=(num_filters, input_channels, filter_size, filter_size))

        return filters

    def forward(self, mini_batch):
        convolution_result, self.cache = TensorMath.convolution(mini_batch, self.filters, self.bias, self.stride, self.padding)
        self.output = self.relu.forward(convolution_result)
        return self.output

    def backprop(self, upstream_grad):
        m = upstream_grad.shape[0]

        # Backward Pass ReLU
        upstream_grad = self.relu.backprop(self.output, upstream_grad)

        # Backward Pass Convolution
        upstream_grad, self.grad_filters, self.grad_bias = TensorMath.backprop_convolution(upstream_grad, self.cache, self.stride, self.padding)
        return upstream_grad

    def update_parameters(self, learning_rate, adam=None):
        if adam is None:
            self.filters -= learning_rate * self.grad_filters
            self.bias -= learning_rate * self.grad_bias
        else:
            parameters = {f"f{self.name}": self.filters, f"b{self.name}": self.bias}
            gradients = {f"f{self.name}": self.grad_filters, f"b{self.name}": self.grad_bias}
            adam.update(parameters, gradients, learning_rate)
            self.filters = parameters[f"f{self.name}"]
            self.bias = parameters[f"b{self.name}"]
