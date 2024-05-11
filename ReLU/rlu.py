import numpy as np

# Define a class implementing the Rectified Linear Unit (ReLU) activation function


class Relu:
    # In the constructor, define the forward and backward pass primitives as two internal operations.
    # You have to implement the forward (fwd) and backward (bwd) pass primitives as lambda expressions.
    def __init__(self):

        # Forward Pass primitive:
        # Forward-propagate only those values of the input batch X, that are > 0
        self.fwd = lambda X: np.maximum(0, X)  # <YOUR CODE HERE> ####

        # Backward Pass primitive
        # Back-propagate only those gradients, for which the corresponding value in X was > 0
        # Hint: you can accomplish the desired objective by multiplying the gradient batch with a boolean mask!
        self.bwd = lambda X, grad: grad * (X > 0)

    # Forward propagation: use the 'fwd' primitive to implement forward propagation
    # Parameters:
    # Input     X:  Batch of input feature maps
    # Output        Batch of activations, i.e. the downsampled input feature maps
    def forward(self, X):
        return self.fwd(X)

    # Backpropagation: use the 'bwd' primitive to implement backprop
    # Parameters:
    # Input     prev_activation:    Previous batch of activations, outputted by the 'forward' function
    # Input     upstream_gradient:  Batch of upstream gradients
    # Output    Back-propagated batch of gradients
    def backprop(self, prev_activation, upstream_gradient):
        return self.bwd(prev_activation, upstream_gradient)


# Test code
np.random.seed(1)

m = 16          # Batch size
c = 3           # Number of channels of the input feature maps
h = 24          # Height of the input feature maps
w = 24          # Width of the input feature maps

# Input feature maps having values in the [-1, +1] range
X = np.random.randn(m, c, h, w) * 2 - 1

# Upstream gradient
dA = np.random.randn(m, c, h, w)

# Create the Relu activation function and use it to get the activation A based on X
relu = Relu()
A = relu.forward(X)

# Backprop the upstream gradient dA and obtain dX
# Formally: dA is the gradient of the cost w.r.t. the input of the next layer
#           dX is the gradient of the cost w.r.t. the input of the ReLU
# Since dX is usually the intermediate output of a convolutional or fully-connected layer, i.e. the non-activated Z, we know that:
# dX (for the ReLU) = dZ (for the conv layer).
# We will use dX to compute the gradient w.r.t the layer's parameters and input.
dX = relu.backprop(A, dA)
print(dX)
