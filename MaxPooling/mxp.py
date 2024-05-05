import numpy as np

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


# Test code
np.random.seed(1)

m = 16          # Batch size
c = 6           # Input feature maps channels
h = 14          # Input feature maps height
w = 14          # Input feature maps width
pool_size = 2   # Pooling window size
stride = 2      # Pooling stride

X = np.random.randn(m, c, h, w)
result = max_pooling(X, pool_size, stride)
output = result["output"]
cache = result["cache"]
