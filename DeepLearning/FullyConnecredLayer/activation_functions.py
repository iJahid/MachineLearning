import numpy as np

# Define a class implementing the Rectified Linear Unit (ReLU) activation function


class Relu:
    # In the constructor, define the forward and backward pass primitives as two internal operations.
    # You have to implement the forward (fwd) and backward (bwd) pass primitives as lambda expressions.
    def __init__(self):

        # Forward Pass primitive:
        # Forward-propagate only those values of the input batch X, that are > 0
        self.fwd = lambda X: np.maximum(0, X)

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

# Realize a class for the numerically-stable Softmax activation function.
# In the lecture we saw how to avoid the numerical overflow in the computation of the exponential.


class Softmax:
    # We want the class to behave like a function:
    # Our objective is to call an instance of the class like we do for functions, for example:
    #
    # Instatiate a new Softmax object:
    # s = Softmax()
    #
    # Use the softmax object like a function:
    # Y_pred = s(X)
    #
    # where X is a mini-batch containing the activations of the last layer.
    #
    # To accomplish this "callable" behavior, we define the __call__ method.
    def __call__(self, X):
        # Compute the numerically-stable exponetial of the input X
        # Remember that X is a mini-batch, so the first axis (0) is the example index
        # The elements of every example are along the second axis (1)
        e_X = np.exp(X - np.max(X, axis=1, keepdims=True))

        # Normalize each exponential example in e_X individually, and return the result
        # The elements of each individual e_x vector must therefore sum up to 1
        return e_X / e_X.sum(axis=1, keepdims=True)
