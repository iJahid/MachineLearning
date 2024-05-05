import numpy as np

# Convolution operation
# Input     X:          Batch of m training examples. Shape: (m, channels, height, width)
# Input     filters:    Filters to convolve with X. Shape: (num_filters, channels, filter_size, filter_size)
# Input     biases:     Biases to add after the convolution. Shape: (num_filters)
# Input     stride:     The stride used to slide the filter on the input images
# Input     padding:    The padding to apply to the input images before the operation
# Output                Dictionary containing the output feature maps and a cache of useful information


def convolution(X, filters, biases, stride, padding):

    # Extract the batch size from the input shape
    m = X.shape[0]  # <YOUR CODE HERE> ####

    # Extract the input example shape in the format: (in_channels, in_height, in_width).
    # We know that X.shape is (m, c, h, w), but we want to extract just (c, h, w).
    # Find a way to extract c, h, and w from X.shape.
    input_shape = X.shape[1:]  # <YOUR CODE HERE> ####

    # Extract the number of filters
    num_filters = filters.shape[0]  # <YOUR CODE HERE> ####

    # Extract the filter size
    # Assume that filters are square, i.e. their width and height are equal.
    # Note: remember that filters have the same number of channels as the input examples. (not relevant now)
    filter_size = filters.shape[2]  # <YOUR CODE HERE> ####

    # Perform zero-padding on the whole input batch X, and save the padded volume in X_pad
    #
    # Note: Use np.pad to add 'padding' zeros before and after each row and each column.
    # We want to pad the rows and the columns of the input images only!
    # Therefore, we must specify to not pad neither in the m direction (training example), nor in the c direction (channels)
    # Since we are using the BDHW (Batch, Depth, Height, Width) data layout, we can positionally specify the directions to pad:
    #
    # For padding an individual direction, e.g. the rows (H), we need to know the padding size before and after each row:
    # (padding_before, padding_after)
    #
    # Therefore, we specify (0, 0) for directions that we don't want to pad. These are B and D.
    # This implies that our final set of padding specifiers is the following:
    #
    # (0, 0), (0, 0), (padding, padding), (padding, padding)
    #
    # This means:
    # - Do not pad the index of the training examples in the batch (B)
    # - Do not pad the channels of the training examples (D)
    # - Pad each row of each training example (H) with 'padding' zeros before and 'padding' zeros after
    # - Pad each column of each training example (W) with 'padding' zeros before and 'padding' zeros after
    #
    # In order to inform np.pad that we want to pad with zeros, we have to specify the string 'constant' ans a parameter.
    # If the constant's value is not explicitly specified, np.pad will automatically pad with the value 0.

    X_pad = np.pad(X, ((0, 0), (0, 0), (padding, padding),
                   (padding, padding)), 'constant')  # <YOUR CODE HERE> ####

    # Compute the output shape: you will have to consider padding!
    # O height =((H+2p-F)/S) =stride
    # O width =((W+2p-F)/S)
    output_height = (input_shape[1] + 2 * padding - filter_size)
    output_width = (input_shape[2] + 2 * padding - filter_size)
    output_shape = (num_filters, output_height, output_width)

    # Initialize the output volume with the correct dimensions, where all elements are initially zero
    out = np.zeros(m, num_filters, output_height,
                   output_width)  # <YOUR CODE HERE> ####

    # Define s = stride and f = filter_size to express the computation in a more compact manner
    s = stride
    f = filter_size

    # Compute the convolution X * filters and store the result in the output volume that you have just created
    # For each input example in the batch:
    for b in range(m):
        # For each output channel: notice how output_channels is equal to num_filters!
        for c in range(num_filters):
            # For each row i of the output
            for i in range(output_height):
                # For each column j of the output
                for j in range(output_width):
                    # Compute the element (b, c, i, j) of the output volume:
                    # To compute the element (c, i, j) of the output, you have to identify the corresponding window in the input X_pad.
                    # Use the "from:to" indexing syntax to identify the correct window in X_pad.
                    # The window of the input to be (element-wise) multiplied with the filters depends on:
                    # i, j, the stride (s) and the filter size (f)
                    # Do not forget to add the bias!
                    out[b, c, i, j] = np.sum(
                        X_pad[b, :, i*s: i*s+f, j*s: j*s+f] * filters[c]) + biases[c]

    # Define some cache that could be useful for future operations, e.g. backprop
    cache = [X, input_shape, output_shape, padding]

    # Return a dictionary with the output and the cache
    return {'output': out, 'cache': cache}


#################################################
# Test code                                     #
#################################################
# Do not modify the seed: it is fixed to 1 in the test-case!
np.random.seed(1)

m = 16              # Batch size
channels = 3        # Input channels
height = 28         # Input height
width = 28          # Input width

padding = 2         # Padding to be applied before the convolution
num_filters = 6     # Number of filters
filter_size = 5     # Filter size (f)
stride = 1          # Filter stride (s)

# Initialize the input volume X
X = np.random.randn(m, channels, height, width)

# Initialize the filters and the biases
filters = np.random.randn(num_filters, channels, filter_size, filter_size)
biases = np.random.randn(num_filters)

# Perform convolution: do not modify the name of the output variables: they are fixed in the test-case

result = convolution(X, filters, biases, stride, padding)
conv_output = result['output']
cache = result['cache']
