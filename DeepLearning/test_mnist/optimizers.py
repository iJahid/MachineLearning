import numpy as np

class AdamOptimizer:
    
    # Constructor
    #
    # Parameters:
    # Input     beta1:      Constant involved in First Moment calculation and correction
    # Input     beta2:      Constant involved in Second Moment calculation and correction
    # Input     epsilon:    Small constant to ensure numerical stability during parameters update
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        # Declare the First and Second Moment volumes
        self.m = None
        self.v = None
        
        # Initialize t = 0 so that it will start counting from 1 (see code below)
        self.t = 0

    # Parameters Update
    #
    # Parameters:
    # Input     parameters:     Dictionary containing the (named) parameters and biases
    # Input     gradients:      Dictionary containing the (named) gradients w.r.t filters and biases
    # Input     learning_rate:  Learing rate to use in the parameter update
    def update(self, parameters, gradients, learning_rate):
        
        # If this 'update' method was not yet called, the First Moment (m) and Second Moment (v) volumes are still 'None'.
        # In this case we should initialize them to empty dictionaries:
        if self.m is None:
            self.m = {}
            self.v = {}

        # Iterate the parameters as (key, value) pairs:
        # The keys are the parameter names, e.g. 'wconv1' for conv1's filters, and 'bconv1' for conv1's biases.
        # The values are the corresponding parameters.
        # Summarizing, for example, the dictionary element 'wconv1' corresponds to conv1's array of filters
        for param_name, param_value in parameters.items():
            
            # If the 'm' and 'v' volumes for the parameter haven't been allocated yet, do it
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param_value)
                self.v[param_name] = np.zeros_like(param_value)

        # Increment t
        self.t += 1
        
        # For each (parameter, gradient) pair:
        for param_name, gradient in gradients.items():
            
            # Compute the First Moment (m) and the Second Moment (v) for the current parameter
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)

            # Compute the First and Second Moment corrections (m^ and v^)
            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            # Update the current parameter 
            parameters[param_name] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            