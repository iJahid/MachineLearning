import numpy as np 
from layer import LayerBase

class OutputLayer(LayerBase):
    
    # Constructor
    #
    # Parameters:
    # Input name:           Name of the layer.
    # Input activation_fn:  Activation function to generate the predictions -- usually a softmax unit.
    # Input cost_fn:        Cost function -- usually Cross-Entropy.
    def __init__(self, name, activation_fn, cost_fn):
        self.activation_fn = activation_fn
        self.cost_fn = cost_fn
        
        super().__init__(name)
        self.type = "output"
        
    # Forward Pass
    #
    # Parameters:
    # Input     X:      Batch of input feature maps
    # Output            Batch of predictions -- Y^
    def forward(self, X):
        
        # Cache the predictions internally
        self.Y_pred = self.activation_fn(X)
        
        # Return the predictions
        return self.Y_pred

    # Get the cost of the last forwarded batch of predictions
    #
    # Parameters:
    # Input     Y:      Batch of labels
    # Output            Cost value -- L
    def get_cost(self, Y):
        
        # Uses the internally cached predictions to compute the cost -- L
        return self.cost_fn(self.Y_pred, Y)

    # Backward Pass
    #
    # Parameters:
    # Input     Y:      Batch of labels
    # Output            Batch containing the gradients of the cost w.r.t. the predictions -- dL/dY^
    def backprop(self, Y):
        
        # Uses the internally cached predictions to compute the gradient of the cost w.r.t. them -- dL/dY^
        return self.cost_fn.gradient(self.Y_pred, Y)