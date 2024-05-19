import numpy as np

class AdamOptimizer:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, parameters, gradients, learning_rate):
        if self.m is None:
            self.m = {}
            self.v = {}

        for param_name, param_value in parameters.items():
            if param_name not in self.m:
                self.m[param_name] = np.zeros_like(param_value)
                self.v[param_name] = np.zeros_like(param_value)

        self.t += 1

        for param_name, gradient in gradients.items():
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * gradient
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (gradient ** 2)

            m_hat = self.m[param_name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param_name] / (1 - self.beta2 ** self.t)

            parameters[param_name] -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)