import numpy as np

class LeakyRelu:

	# Constructor
    # Parameters:
    # Input     alpha:	Factor used in the Leaky ReLU formula
    def __init__(self, alpha=0.01):
        self.alpha = alpha

	# Forward Pass
    # Parameters:
    # Input     X:  Batch of input feature maps
    # Output        Batch of activations, i.e. the downsampled input feature maps
    def forward(self, X):
        return np.where(X >= 0, X, self.alpha * X)

	# Backward Pass:
	# Back-propagate the full value of those gradients, for which the corresponding value in X was >= 0.
	# Otherwise, back-propagate a factor 'alpha' of the gradient.
	# Hint: You can use the numpy 'where' function on the previous activation to determine where X was >= 0.
	#
	# Parameters:
    # Input     prev_activation:    Previous batch of activations, outputted by the 'forward' function
    # Input     upstream_gradient:  Batch of upstream gradients
    # Output    Back-propagated batch of gradients
    def backprop(self, prev_activation, upstream_gradient):
        return np.where(prev_activation >= 0, upstream_gradient, self.alpha * upstream_gradient)
