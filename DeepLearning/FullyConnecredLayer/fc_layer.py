import numpy as np
from operations import convolution
from activation_functions import Relu
from layer import LayerBase
from persistence import read_parameters, write_parameters


class FullyConnectedLayer(LayerBase):

    # Constructor
    #
    # Parameters:
    # Input name:           Name of the layer
    # Input input_shape:    Shape of the input feature maps -- can be either 1D, 2D or 3D
    # Input output_neurons: Number of output neurons -- the output for one input feature map is 1D
    def __init__(self, name, input_shape, output_neurons):
        self.input_shape = input_shape
        self.output_shape = output_neurons

        # Define the number of input neurons based on the shape of the input feature maps
        input_neurons = np.prod(input_shape)

        # Initialize the weights matrix (W) randomly, with normally-distributed values in [0, 1]
        # Be careful: in the theory, we have defined W as a (output_neurons, input_neurons) matrix.
        # This works because we consider our input features x(i) as column-vectors.
        # Therefore, in the theory, the batch X is a (input_neurons, m) matrix.
        # This allows us to perform W*X as a (output_neurons, input_neurons)*(input_neurons, m) operation:
        # As a result, we get u = (W*X) as a (output_neurons, m) matrix.
        # At this point, we add the (output_neurons, 1) bias vector to every vector of the batch 'u': y = u + b.
        #
        # However, in numpy, the (flattened) input batch X is of shape (m, input_neurons).
        # This implies that out x(i) feature maps are 'm' row-vectors of size 'input_neurons'.
        # We must preform the matrix-vector multiplication between W and X in such a way, that the result matches the output shape (m, output_neurons)
        # We can accomplish this in the following way:
        #
        # X is a (m, input_neurons) matrix.
        # Let W be a (input_neurons, output_neurons) matrix.
        # In order to get a (m, output_neurons) result with the dot product, we must compute X*W:
        # This is a (m, input_neurons)*(input_neurons, output_neurons) operation, resulting in a (m, output_neurons) matrix.
        #
        # In conclusion, we can initialize W as a (input_neurons, output_neurons) matrix and perform X*W.
        self.weights = np.random.randn(input_neurons, output_neurons)

        # Initialize the bias vector to be of size 'output_neurons'
        self.biases = np.zeros(output_neurons)

        # Use the ReLU activation function that you have implemented in a previous exercise
        self.activation = Relu()

        # Call the constructor of the ancestor class
        super().__init__(name)

    def get_parameters(self):
        return {
            "layer_name": self.name,
            "weights": self.weights,
            "biases": self.biases
        }

    def set_parameters(self, parameters):
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

    # ---- YOUR TASK 1) STARTS HERE ----

    # Persist the parameters of this layer to disk
    #
    # Parameters:
    # Input     folder:     Destination folder into which the parameters will be stored.
    def persist(self, folder):

        # Use the 'get_parameters' function to retrieve a ready-to-use dictionary containing the parameters
        parameters = self.get_parameters()

        # Store the parameters using the persistence functions which you implemented in the previous exercise
        write_parameters(parameters, folder)

    # Load the parameters of this layer from disk
    #
    # Parameters:
    # Input     folder:     Source folder from which the parameters will be loaded.
    def load(self, folder):

        # Load the parameters using the persistence functions which you implemented in the previous exercise
        parameters = read_parameters(folder, self.name)

        # Use the 'set_parameters' function to set the loaded parameters
        self.set_parameters(parameters)

    # ---- YOUR TASK 1) ENDS HERE ----

    # Forward Pass
    #
    # Parameters:
    # Input X:  Batch of feature maps -- each feature map can be either 1D, 2D or 3D
    # Output    Batch of activations of size (output_neurons) each
    def forward(self, X):
        # Store the batch of input feature maps
        self.X = X

        # Flatten the input feature maps in order to obtain a (m, input_neurons) matrix
        # Originally, each one of the 'm' feature maps is of shape (c, h, w).
        # We have to flatten them into row-vectors of c*h*w elements, where c*h*w = input_neurons.
        # This flattening operation allows for the dot product: X*W.
        # Use the 'reshape' method on X, but ensure to keep the batch-index dimension unchanged:
        # This way, we obtain 'm' separate vectors of size c*h*w each.
        # Hint: to keep a specific dimension, pass -1 in the corresponding parameter of the 'reshape' function.
        self.flat_X = X.reshape(-1, np.prod(self.input_shape))

        # Compute the intermediate output Z = X*W + b
        Z = np.dot(self.flat_X, self.weights) + self.biases

        # Compute the activation of 'Z', store it to the internal variable 'prev_activation', and return it.
        # The internal variable 'prev_activation' contains, at any time, the last activation of the layer.
        # The last activation is used later to back-propagate the upstream gradient through the activation function.
        A = self.activation.forward(Z)
        self.prev_activation = A
        return A

    # Backward Pass
    #
    # Parameters:
    # Input upstream_grad:  Batch of upstream gradients w.r.t. the activation of this layer -- dL/dA
    # Output                Batch of gradients w.r.t. the input of this layer -- dL/dX
    def backprop(self, upstream_grad):

        # Extract the batch size 'm'
        m = upstream_grad.shape[0]

        # Perform the backprop of the upstream gradient (dL/dA) through the activation unit
        # From this first backprop step, we obtain dL/dZ: we call it 'dZ'
        dZ = self.activation.backprop(self.prev_activation, upstream_grad)

        # Compute the gradient of the cost w.r.t. the weights W. This is what we call dL/dW.
        # Using the Chain Rule of differentiation, we can decompose dL/dW into dL/dZ * dZ/dW.
        # As we saw in the theory, we can compute dZ/dW very easily.
        # Warning! This is a batch of gradients: see the theory of "Backward Pass in Fully-Connected Layers".
        # Hint: you must use np.dot and transpose one of its arguments.
        self.grad_weights = np.dot(self.flat_X.T, dZ) / m

        # Compute the gradient of the cost w.r.t. the biases b. This is what we call dL/db.
        # Thanks to the Chain Rule, dL/db can be decomposed into dL/dZ * dZ/db.
        # Compute dZ/db and use the above information to compute dL/db.
        # Warning! This is a batch of gradients: see the theory of "Backward Pass in Fully-Connected Layers".
        self.grad_biases = np.sum(dZ, axis=0) / m

        # Compute the gradient of the cost w.r.t. the input batch X. This is what we call dL/dX.
        # The Chain Rule allows us to decompose dL/dX into dL/dZ * dZ/dX
        # Check the theory of "Backward Pass in Fully-Connected Layers" and find out how to compute dZ/dX.
        # Use dZ/dX and the above information to compute dL/dX, which we call 'dX' for simplicity.
        dX = np.dot(dZ, self.weights.T)

        # Be careful:
        # dZ is of shape (m, output_neurons)
        # weights is of shape (input_neurons, output_neurons)
        # Dimensionally, dZ * weights.T is: (m, output_neurons) * (output_neurons, input_neurons)
        # The result of this operation is of dimension (m, input_neurons)
        # Therefore, we have obtained a batch of 'm' flattened gradients dL/dX.
        # We must restore their original dimensions:
        # - The gradients are already organized into 'm' different vectors, which is correct.
        # - However, each one of these vectors must be reshaped to the same dimensions of the inputs x(i).
        # Let's consider the whole input batch X: its size was (m, c, h, w).
        # If we reshape the whole dL/dX batch to (m, c, h, w), the first dimension (batch index) is kept, and the elements of each vector dL/dX(i) are reshaped to (c, h, w) individually.
        # As a result, the dL/dX batch has the same shape of the input batch X.
        #
        # Note: we have to reshape dL/dX only if the input examples x(i) were 3D
        if type(self.input_shape) is tuple:
            dX = dX.reshape(m, *self.input_shape)

        # Return dL/dX to back-propagate the gradients
        return dX

    # Update Parameters:
    # This procedure is used to update the weights and the biases based on their gradients: dL/dW and dL/db, respectively.
    #
    # Parameters:
    # Input     learning_rate:  Learning rate used for Gradient Descent
    def update_parameters(self, learning_rate):

        # ---- YOUR TASK 2) STARTS HERE ----

        if not self.frozen:
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases

        # ---- YOUR TASK 2) ENDS HERE ----
