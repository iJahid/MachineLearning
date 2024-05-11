import numpy as np

# Convolution operation
# Input     X:          Batch of m training examples. Shape: (m, channels, height, width)
# Input     filters:    Filters to convolve with X. Shape: (num_filters, channels, filter_size, filter_size)
# Input     biases:     Biases to add after the convolution. Shape: (num_filters)
# Input     stride:     The stride used to slide the filter on the input images
# Input     padding:    The padding to apply to the input images before the operation
# Output                Dictionary containing the output feature maps and a cache of the input information


def convolution(X, filters, biases, stride, padding):

    # Extract the batch size from the input shape
    m = X.shape[0]

    # Extract the input example shape in the format: (in_channels, in_height, in_width).
    # We know that X.shape is (m, c, h, w), but we want to extract just (c, h, w).
    # Find a way to extract c, h, and w from X.shape.
    input_shape = X.shape[1:]

    # Extract the number of filters
    num_filters = filters.shape[0]

    # Extract the filter size
    # Assume that filters are square, i.e. their width and height are equal.
    # Note: remember that filters have the same number of channels as the input examples. (not relevant now)
    filter_size = filters.shape[2]

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
                   (padding, padding)), 'constant')

    # Compute the output shape: you will have to consider padding!
    output_height = (input_shape[1] + 2 * padding - filter_size) // stride + 1
    output_width = (input_shape[2] + 2 * padding - filter_size) // stride + 1
    output_shape = (num_filters, output_height, output_width)

    # Initialize the output volume with the correct dimensions, where all elements are initially zero
    out = np.zeros((m, num_filters, output_height, output_width))

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
    cache = {'X': X, 'X_pad': X_pad, 'input_shape': input_shape,
             'output_shape': output_shape, 'padding': padding}

    # Return a dictionary with the output and the cache
    return {'output': out, 'cache': cache}

# Max Pooling operation
#
# Parameters:
# Input     X:          Batch of m feature maps, each of size (in_c, in_h, in_w)
# Input     pool_size:  Pooling window size
# Input     stride:     The pooling stride that we use to slide the pooling window over the input feature map
# Output    Tuple containing the output downsampled feature map, and a cache of useful information


def max_pooling(X, pool_size, stride):

    # Retrieve the dimensions of the input volume X
    (m, in_c, in_h, in_w) = X.shape

    # Calculate the dimensions of the output volume
    # Note: pooling is applied to each channel of the input feature maps independently
    # Therefore, the height and width decrease, but the number of channels does not vary
    out_c = in_c
    out_h = int(1 + (in_h - pool_size) / stride)
    out_w = int(1 + (in_w - pool_size) / stride)

    # Initialize output volume with zeros
    output = np.zeros((m, out_c, out_h, out_w))

    for i in range(m):  # loop over the input feature maps
        for c in range(out_c):  # loop over the channels of the output volume
            for h in range(out_h):  # loop on the vertical axis of the output volume
                for w in range(out_w):  # loop on the horizontal axis of the output volume

                    # Identify the current window (2D) of the i-th input feature map
                    # The window is defined by a vertical range (h_start -> h_end) and an horizontal range (w_start -> w_end)
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size

                    # Extract the window of the current feature map (i), in the current channel (c)
                    input_window_2D = X[i, c, h_start:h_end, w_start:w_end]

                    # The output value located at [i, c, h, w] is the maximum value of the window
                    output[i, c, h, w] = np.max(input_window_2D)

    # Cache the input, the pooling size and the stride
    # These information will be useful during backprop
    cache = (X, pool_size, stride)

    return {"output": output, "cache": cache}

# Max Pooling operation
#
# Parameters:
# Input     X:          Batch of m feature maps, each of size (in_c, in_h, in_w)
# Input     pool_size:  Pooling window size
# Input     stride:     The pooling stride that we use to slide the pooling window over the input feature map
# Output    Tuple containing the output downsampled feature map, and a cache of useful information


def max_pooling(X, pool_size, stride):

    # Retrieve the dimensions of the input volume X
    (m, in_c, in_h, in_w) = X.shape

    # Calculate the dimensions of the output volume
    # Note: pooling is applied to each channel of the input feature maps independently
    # Therefore, the height and width decrease, but the number of channels does not vary
    out_c = in_c
    out_h = int(1 + (in_h - pool_size) / stride)
    out_w = int(1 + (in_w - pool_size) / stride)

    # Initialize output volume with zeros
    output = np.zeros((m, out_c, out_h, out_w))

    for i in range(m):  # loop over the input feature maps
        for c in range(out_c):  # loop over the channels of the output volume
            for h in range(out_h):  # loop on the vertical axis of the output volume
                for w in range(out_w):  # loop on the horizontal axis of the output volume

                    # Identify the current window (2D) of the i-th input feature map
                    # The window is defined by a vertical range (h_start -> h_end) and an horizontal range (w_start -> w_end)
                    h_start = h * stride
                    h_end = h_start + pool_size
                    w_start = w * stride
                    w_end = w_start + pool_size

                    # Extract the window of the current feature map (i), in the current channel (c)
                    input_window_2D = X[i, c, h_start:h_end, w_start:w_end]

                    # The output value located at [i, c, h, w] is the maximum value of the window
                    output[i, c, h, w] = np.max(input_window_2D)

    # Cache the input, the pooling size and the stride
    # These information will be useful during backprop
    cache = (X, pool_size, stride)

    return {"output": output, "cache": cache}


# Max Pooling Backprop
#
# Parameters:
# Input     upstream_grad:  Batch containing the gradients of the cost w.r.t the outputs of the pooling layer (dL/dA)
# Input     cache:          The cache returned by the Max Pooling forward pass
# Output    Batch containing the gradients of the cost w.r.t. the input feature maps (dL/dX)
def max_pooling_backprop(upstream_grad, cache):

    # Retrieve the input batch X, the pooling size and the stride from the cache
    (X, pool_size, stride) = cache

    # Retrieve the input dimensions from the shape of X
    m, input_channels, input_height, input_width = X.shape

    # Retrieve the output dimensions from the shape of dL/dA
    m, output_channels, output_height, output_width = upstream_grad.shape

    # Optional: you could assert that the number of input and output channels must be the same
    assert (output_channels == input_channels)

    # Initialize the gradient w.r.t the input (dL/dX) with zeros: it has the same dimensions as the input batch X
    input_grad = np.zeros(X.shape)

    for i in range(m):  # loop over the examples in the batch
        for c in range(output_channels):  # loop over the output channels
            for h in range(output_height):  # loop over the height of the output volume
                for w in range(output_width):  # loop over the width of the output volume
                    # Identify the current 2D window of channel 'c' of the input example.
                    # The 2D window is of size (pool_size, pool_size).
                    h_start = h
                    h_end = h_start + pool_size
                    w_start = w
                    w_end = w_start + pool_size

                    # Extract the 2D window
                    example_window_2D = X[i, c, h_start:h_end, w_start:w_end]

                    # Create the maximum mask of the current window:
                    # The maximum mask is a boolean matrix of the same size as the current window.
                    # In the maximum mask, the element corresponding to the current window's maximum value is set to True.
                    # The rest is set to False.
                    # Hint: use np.max to find the maximum value in the window, and combine it with the '==' operator
                    mask = example_window_2D == np.max(example_window_2D)

                    # Identify the current entry of the upstream gradient:
                    # Hint: it is located in the i-th upstream gradient of the batch, at position (c, h, w)
                    d = upstream_grad[i, c, h, w]

                    # According to the Backprop theory for Max Pooling, compute the gradient w.r.t. the input (dL/dX):
                    # 1. Multiply the current gradient entry with the maximum mask to determine where it should be back-propagated.
                    # 2. Add the result to the current window of the input gradient.
                    input_grad[i, c, h_start:h_end,
                               w_start:w_end] += np.multiply(mask, d)

    # Return the batch containing the gradients of the cost w.r.t. the input feature maps.
    # This is dL/dX.
    return input_grad
