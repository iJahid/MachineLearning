from conv_layer import ConvolutionalLayer
from output_layer import OutputLayer
import numpy as np

# Perform one training step:
# This procedure takes a model, batch of training examples X and a batch of corresponding labels Y.
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
# Input     model:              Model containing the list of all layers of the convolutional neural network
# Input     X:                  Batch of training examples
# Input     Y:                  Batch of labels
# Input     alpha:              Learning rate
# Input     use_adam:           Whether to use the Adam optimizer or plain BGD
# Input     update_parameters:  Optionally specify whether to perform parameters update. Default value: True
# Output                        Current cost
def train(model, X, Y, alpha, use_adam, update_parameters=True):

    # Forward Pass:
    # The layers share the common interface inherited by LayerBase.
    # Therefore we can perform forward / backprop by iterating the list and calling the same methods uniformly each element.
    # Initialize A = X to start the forward propagation
    A = X
    for i, layer in enumerate(model.layers):
        A = layer.forward(A)
        if isinstance(layer, ConvolutionalLayer) and isinstance(model.layers[i + 1], OutputLayer) and layer.use_as_fc:
            A = np.squeeze(A)

    # At this point, A is the activation of the output layer: therefore, A contains the predictions Y^.
    # Compute the cost of the current predictions
    L = model.layers[-1].get_cost(Y)

    # Backward Pass:
    # The output layer has saved the predictions Y^ in its internal cache.
    # In order to start backprop from the output layer, we have to provide it with the labels Y.
    # Therefore, we initialize dA to be initially Y. 
    # At the end of the backprop process, dA will contain dL/dX for the first layer of the network.
    dA = Y
    for i, layer in enumerate(reversed(model.layers)):
        reversed_i = len(model.layers) - 1 - i
        dA = layer.backprop(dA)

        if isinstance(layer, OutputLayer) and isinstance(model.layers[reversed_i - 1], ConvolutionalLayer) and model.layers[reversed_i - 1].use_as_fc:
            dA = np.expand_dims(dA, axis=2)
            dA = np.expand_dims(dA, axis=3)

    # Update the parameters of all layers based on the computed gradients
    if update_parameters:
        for layer in model.layers:
            layer.update_parameters(alpha, adam=use_adam)

    # Return the current cost
    return L
