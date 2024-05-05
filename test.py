import numpy as np

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
        e_X = np.exp(X-X.max())
        sum_ex = sum(e_X)
        # Normalize each exponential example in e_X individually, and return the result
        # The elements of each individual e_x vector must therefore sum up to 1
        return e_X/sum_ex


# Test the numerically-stable Softmax activation
m = 32  # batch size
num_classes = 10  # number of output classes: this is the dimension of one prediction y^
np.random.seed(1)
X = np.random.randn(m, num_classes)  # batch of 32 examples of size 10 each

# Instatiate the Softmax unit and call it
s = Softmax()
Y_pred = s(X)
print(Y_pred)
