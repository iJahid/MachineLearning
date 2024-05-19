import numpy as np

# Perform one training step:
# This procedure takes a convnet, batch of training examples X and a batch of corresponding labels Y.
# The single training step implemented below consists of the following actions:
# 1. Perform the Forward Pass and obtain the predictions -- Y^
# 2. Compute the cost of the current predictions -- L
# 3. Compute the gradient of the cost w.r.t. the current predictions -- dL/dY^
# 4. Backprop dL/dY^ through the network and compute the gradients w.r.t. the parameters of all layers
# 5. Update the parameters of all layers using the computed gradients
# 6. Return the current cost -- L
#
# Parameters:
#
# Input     convnet:            List containing all the layers of the convolutional neural network
# Input     X:                  Batch of training examples
# Input     Y:                  Batch of labels
# Input     alpha:              Learning rate
# Input     update_parameters:  Optionally specify whether to perform parameters update. Default value: True
# Output                Current cost


def train(convnet, X, Y, alpha, update_parameters=True):

    # Forward Pass:
    # The layers share the common interface inherited by LayerBase.
    # Therefore we can perform forward / backprop by iterating the list and calling the same methods uniformly each element.
    # Initialize A = X to start the forward propagation
    A = X
    for layer in convnet:
        A = layer.forward(A)

    # At this point, A is the activation of the output layer: therefore, A contains the predictions Y^.
    # Compute the cost of the current predictions
    L = convnet[-1].get_cost(Y)

    # Backward Pass:
    # The output layer has saved the predictions Y^ in its internal cache.
    # In order to start backprop from the output layer, we have to provide it with the labels Y.
    # Therefore, we initialize dA to be initially Y.
    # At the end of the backprop process, dA will contain dL/dX for the first layer of the network.
    dA = Y
    for layer in reversed(convnet):
        dA = layer.backprop(dA)

    # Update the parameters of all layers based on the computed gradients
    if update_parameters:
        for layer in convnet:
            layer.update_parameters(alpha)

    # Return the current cost
    return L
