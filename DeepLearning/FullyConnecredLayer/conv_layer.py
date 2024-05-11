import numpy as np
from operations import convolution
from activation_functions import Relu
from layer import LayerBase


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

        # Initialize to zeros the gradient of the cost w.r.t. the filters: dL/dW -- let's call it 'grad_filters'
        # Note: dL/dW (grad_filters) must have the same dimensions of the filters volume
        self.grad_filters = np.zeros_like(self.filters)

        # Initialize to zeros the gradient of the cost w.r.t. the biases: dL/db -- let's call it 'grad_biases'
        # Note: dL/db (grad_biases) must have the same dimensions of the biases vector
        self.grad_biases = np.zeros_like(self.biases)

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
        s = self.stride
        f = self.filter_size

        for b in range(m):  # loop over the input examples of the batch
            # loop over the number of filters (= output channels)
            for c in range(self.num_filters):
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
                        grad_input[b, :, i*s: i*s+f, j*s: j*s +
                                   f] += self.filters[c] * dZ[b, c, i, j]

                        # According to the backprop formulas, the gradient w.r.t. a specific filter is computed as follows:
                        # First, we have to identify the current 3D window of the padded input X_pad that was used in the Forward Pass.
                        # The current 3D window of X_pad can be found in the same way we did for dX.
                        # We multiply the 3D window of X_pad with the current upstream gradient element dZ(b, c, i, j)
                        # Note: dZ(b, c, i, j) is a scalar (a number)
                        # Then, we add the result to the gradient for the current filter.
                        self.grad_filters[c] += X_pad[b, :, i *
                                                      s: i*s+f, j*s: j*s+f] * dZ[b, c, i, j]

                # According to the backprop formulas, the gradient w.r.t. a specific bias is computed as follows:
                # First, we have to take the 2D slice of the upstream gradient (dZ) identified by the current batch-index b and the current output channel c.
                # Then, we have to sum all the elements of this 2D slice, obtaining a scalar value (i.e., a number).
                # Finally, we add the obtained scalar to the current bias.
                # Note: since the number of biases is equal to the number of filters, the current bias is identified by 'c'.
                # Be careful: according to the backprop formulas, since we are summing entire 2D windows of the upstream gradient,
                # and we are doing this for each example b of the batch, the magnitude of the gradients w.r.t. the biases is kind of
                # multiplied by a factor 'm' (batch size). Therefore, when we are finished summing the contributing 2D windows,
                # we must divide dL/db by m to get the mean bias gradient over the batch.
                self.grad_biases[c] += np.sum(dZ[b, c])

        # Compute the mean dL/db over the batch
        self.grad_biases /= m

        # At this point, the gradients dL/dW, dL/b and dL/dX have been computed.
        # There is still one step to do: for the calculations, we had to consider the padded input 'X_pad'.
        # As a result, the height and width of dL/dX (grad_input) might be larger than the original height and width of X!
        # The gradient elements that we have computed for the padding boundary turn out to be useless information.
        # We can easily get rid of them: use range indexing to exclude the padding and return the significant part of 'grad_input'
        # The significant part of grad_input must have the same dimensions of the original input batch X.
        # Use the internal variable 'input_shape' to retrieve the original height and width of X.
        return grad_input[:, :, self.padding:self.input_shape[1] + self.padding, self.padding:self.input_shape[2] + self.padding]

    # Update Parameters:
    # This procedure is used to update the filters and the biases based on their gradients: dL/dW and dL/db, respectively.
    #
    # Parameters:
    # Input     learning_rate:  Learning rate used for Gradient Descent
    def update_parameters(self, learning_rate):

        # Update the filters and the biases according to the Gradient Descent formulas:
        self.filters -= learning_rate * self.grad_filters
        self.biases -= learning_rate * self.grad_biases
