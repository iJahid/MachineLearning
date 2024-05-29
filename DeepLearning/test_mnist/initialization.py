import numpy as np

# He Initialization -- Convolutional Filters
#
# This procedure initializes an array of convolutional filters of the requested size with He Initialization.
# This procedure is designed to be used during the setup of convolutional layers. 
#
# Parameters:
# Input     num_filters:    Number of filters to create 
# Input     in_channels:    Number of channels of the filters -- It is equal to the convolutional layer's input depth
# Input     filter_size:    Filter size -- Filters are square
# Output    filters:        Array of He-initialized filters
def he_initialization_filters(num_filters, in_channels, filter_size):

    # Initialize a new array of filters with He
    filters = np.random.randn(
        num_filters,
        in_channels,
        filter_size,
        filter_size
    ) * np.sqrt(2. / (in_channels * filter_size * filter_size))
    
    # Return the initialized filters
    return filters 
    
# He Initialization -- Fully Connected Weight Matrix
#
# This procedure initializes a weight matrix of the requested size with He Initialization.
# This procedure is designed to be used during the setup of fully connected layers. 
#
# Parameters:
# Input     input_shape:    Potentially 3D input shape -- If the previous layer is a conv layer, the input shape is 3D.
#                           Otherwise, if the previous layer is a fully connected layer, the input shape is 1D.
# Input     output_shape:   Number of output neurons
# Output    weights:        He-initialized weight matrix
def he_initialization_weights(input_shape, output_shape):
    
    # We know that fully connected layers perform 2D matrix-vector multiplication + bias addition.
    # Therefore, we should handle the case in which the input shape is 3D:
    # In this case, the 3D input to the fully connected layer will be flattened into a 1D vector of length 'input_neurons'.
    # Compute the number of input neurons, independently on the dimensionality of the input shape (1D or 3D):
    input_neurons = np.prod(input_shape)
    
    # Initialize a new weight matrix with He
    weights = np.random.randn(input_neurons, output_shape) * np.sqrt(2. / input_neurons)
    
    # Return the initialized weight matrix 
    return weights 
    