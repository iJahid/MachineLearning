from output_layer import OutputLayer
import numpy as np

# Perform one Testing / Inference step:
# This procedure takes a model, batch of examples X and, optionally, the batch size m if m > 1.
# The single Testing / Inference step implemented below consists of the following actions:
# 1. Perform the Forward Pass and obtain the predictions -- Y^
# 2. Return the predictions -- Y^
#
# Parameters:
#
# Input     model:              Model containing the list of all layers of the convolutional neural network
# Input     X:                  Batch of testing examples
# Input     m:                  Optional batch size, if you are using a batch size m > 1
# Output                        Predictions Y^
def forward(model, X, m=1):
    # Forward Pass:
    # The layers share the common interface inherited by LayerBase.
    # Therefore we can perform the forward pass by iterating the list and calling the same methods uniformly each element.
    # Initialize A = X to start the forward propagation
    A = X
    for i, layer in enumerate(model.layers):
        A = layer.forward(A)

        if m == 1 and i < len(model.layers) - 1 and isinstance(model.layers[i + 1], OutputLayer):
            # The batch size is m = 1, so we need that the first dimension is of length m = 1
            A = A.reshape(1, -1)

    return A
