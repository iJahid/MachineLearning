import numpy as np


class CrossEntropyCost:

    # Define a __call__ method allowing to call CrossEntropyCost objects like if they were functions
    #
    # Parameters:
    #
    # Input Y_pred: batch of m predictions
    # Input Y_true: batch of m labels
    # Output:       cost value computed over the whole batch
    def __call__(self, Y_pred, Y_true):

        # Define a small epsilon value to guarantee numerical stability
        # You will have to find out where you need to use epsilon
        epsilon = 1e-10

        # Extract the batch size m
        m = Y_pred.shape[0]

        # Compute the cost over the batch
        cost = -np.sum(Y_true*np.log(Y_pred+epsilon)) / \
            m  # -sum(yi,logY1)/m ####

        # Return the computed cost
        return cost

    # Compute the gradient of the cost with respect to the prediction batch
    #
    # Input Y_pred: batch of m predictions
    # Input Y_true: batch of m labels
    # Output:       gradient of the cost w.r.t. the prediction, computed over the whole batch
    #
    # Note: the gradient over the whole batch is the mean gradient of all the dL/dy^(i)
    def gradient(self, Y_pred, Y_true):

        # Extract the batch size m
        m = Y_pred.shape[0]

        # Compute the Softmax-CrossEntropy gradient with respect to the predictions
        # Note: this is the gradient that we will decompose into the contributions of all the parameters of the network
        # Backprop takes this gradient and decomposes is into individual gradients w.r.t. the weights and biases of each layer
        return (Y_pred-Y_true)/m


# Test code
# You don't need to modify anything below this line. If you desire, you can tune 'm' and 'num_classes'.
# Do not modify the random seed: the seed value is fixed to 1 in the test-case!
np.random.seed(1)
m = 16
num_classes = 25

# Simulate a batch of one-hot labels Y
Y = np.zeros((m, num_classes))
for i in range(m):
    Y[i, np.random.randint(num_classes)] = 1

# Simulate a batch of predictions Y_pred
Y_pred = np.random.randn(m, num_classes)
Y_pred = Y_pred / Y_pred.sum(axis=1, keepdims=True)

# Create a Cross-Entropy cost function
cost_function = CrossEntropyCost()

# Compute the cost over the whole batch
cost = cost_function(Y_pred, Y)

# Compute the gradient of the cost w.r.t the prediction, over the whole batch
grad = cost_function.gradient(Y_pred, Y)
