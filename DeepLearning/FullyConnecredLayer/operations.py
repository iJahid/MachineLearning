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

# Convolution Backprop
# This procedure computes the gradients of the cost w.r.t. the filters (dL/dW), the biases (dL/db), and the input batch (dL/dX).
# The gradients of the cost w.r.t. the input batch, dL/dX, is returned as an output to continue the Backpropagation.
# Summarizing, the following gradients are returned in the order: dL/dW, dL/db, dL/dX.
#
# Parameters:
# Input     input_shape:    Input shape of the feature maps -- The format of the input shape is (c, h, w)
# Input     filters:        Filters used in the convolution Forward Pass
# Input     stride:         Stride used in the convolution Forward Pass
# Input     biases:         Biases vector
# Input     dZ:             dL/dZ -- The upstream gradients dL/dA back-propagated through the activation unit
# Input     X_pad:          Batch of the padded input feature maps X
# Input     padding:        Thickness of the zero-padding boundary applied to the feature maps in X
# Input     upstream_grad:  Upstream batch containing the gradients of the cost w.r.t. the activation of this layer: dL/dA
# Output    The gradients of the cost w.r.t. the filters -- dL/dW
# Output    The gradients of the cost w.r.t. the biases -- dL/db
# Output    The gradients of the cost w.r.t. the input batch X: dL/dX -- this can be seen as dL/dA for the previous layer.


def convolution_backprop(input_shape, filters, stride, biases, dZ, X_pad, padding, upstream_grad):

    # Extract the number of filters
    num_filters = filters.shape[0]

    # Extract the filter size
    filter_size = filters.shape[2]

    # Initialize to zeros the gradient of the cost w.r.t. the filters: dL/dW -- let's call it 'grad_filters'
    # Note: dL/dW (grad_filters) must have the same dimensions of the filters volume
    grad_filters = np.zeros_like(filters)

    # Initialize to zeros the gradient of the cost w.r.t. the biases: dL/db -- let's call it 'grad_biases'
    # Note: dL/db (grad_biases) must have the same dimensions of the biases vector
    grad_biases = np.zeros_like(biases)

    # Initialize to zeros the gradient of the cost w.r.t. the input X of this layer: dL/dX -- let's call it 'grad_input'
    # Note: dL/dX (grad_input) must have the same dimensions of the original input volume X
    # However, for the intermediate calculations, it is fundamental to consider the padding which was applied X!
    grad_input = np.zeros_like(X_pad)

    # In order to compute the gradients, we need to extract the batch size 'm', the output height and the output width.
    # Hint: the output shape was saved in the cache, but we could also read it from 'upstream_gradient' (dA) or from 'dZ', because:
    # - The dimensions of the layer's activation (A) are the same of 'upstream_gradient' (dA).
    # - The dimensions of dA are the same of the intermediate gradient dZ (which we consider the new upstream gradient).
    m = dZ.shape[0]
    out_h = dZ.shape[2]
    out_w = dZ.shape[3]

    # Define some convenient variables with short names to simplify formulas
    s = stride
    f = filter_size

    for b in range(m):  # loop over the input examples of the batch
        for c in range(num_filters):  # loop over the number of filters (= output channels)
            for i in range(out_h):  # loop over the rows of the output volume
                for j in range(out_w):  # loop over the columns of the output volume
                    # Identify the current 3D window of the input gradient dX:
                    # The current 3D window of the input gradient dX is defined by (i, j), the stride s, and the filter size f.
                    # Note: the current 3D window of dX has the same dimensions of a filter.
                    # According to the backprop formulas, we must do the following:
                    # 1. Multiply the current filter by the current upstream gradient element dZ(b, c, i, j), which is a scalar.
                    # 2. Take the result of 1. and add it to the 3D window of the input gradient dX.
                    # Note: the 3D window of dX and the filter multiplied by the upstream gradient element dZ(b, c, i, j) have the same dimensions. Therefore, we can add them together.
                    # Hint: use the colon indexer (:) to select all channels of the 3D window of dX (grad_input)
                    grad_input[b, :, i*s: i*s+f, j*s: j *
                               s+f] += filters[c] * dZ[b, c, i, j]

                    # According to the backprop formulas, the gradient w.r.t. a specific filter is computed as follows:
                    # First, we have to identify the current 3D window of the padded input X_pad that was used in the Forward Pass.
                    # The current 3D window of X_pad can be found in the same way we did for dX.
                    # We multiply the 3D window of X_pad with the current upstream gradient element dZ(b, c, i, j)
                    # Note: dZ(b, c, i, j) is a scalar (a number)
                    # Then, we add the result to the gradient for the current filter.
                    grad_filters[c] += X_pad[b, :, i*s: i *
                                             s+f, j*s: j*s+f] * dZ[b, c, i, j]

            # According to the backprop formulas, the gradient w.r.t. a specific bias is computed as follows:
            # First, we have to take the 2D slice of the upstream gradient (dZ) identified by the current batch-index b and the current output channel c.
            # Then, we have to sum all the elements of this 2D slice, obtaining a scalar value (i.e., a number).
            # Finally, we add the obtained scalar to the current bias.
            # Note: since the number of biases is equal to the number of filters, the current bias is identified by 'c'.
            # Be careful: according to the backprop formulas, since we are summing entire 2D windows of the upstream gradient,
            # and we are doing this for each example b of the batch, the magnitude of the gradients w.r.t. the biases is kind of
            # multiplied by a factor 'm' (batch size). Therefore, when we are finished summing the contributing 2D windows,
            # we must divide dL/db by m to get the mean bias gradient over the batch.
            grad_biases[c] += np.sum(dZ[b, c])

    # Compute the mean dL/db over the batch
    grad_biases /= m

    # At this point, the gradients dL/dW, dL/b and dL/dX have been computed.
    # There is still one step to do: for the calculations, we had to consider the padded input 'X_pad'.
    # As a result, the height and width of dL/dX (grad_input) might be larger than the original height and width of X!
    # The gradient elements that we have computed for the padding boundary turn out to be useless information.
    # We can easily get rid of them: use range indexing to exclude the padding and return the significant part of 'grad_input'
    # The significant part of grad_input must have the same dimensions of the original input batch X.
    # Use the internal variable 'input_shape' to retrieve the original height and width of X.
    return grad_filters, grad_biases, grad_input[:, :, padding:input_shape[1] + padding, padding:input_shape[2] + padding]

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
