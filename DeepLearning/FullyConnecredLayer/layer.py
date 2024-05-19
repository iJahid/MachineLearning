class LayerBase:
    def __init__(self, name):
        self.name = name
        self.frozen = False

    # ---- YOUR TASK 2) STARTS HERE ----

    # Freeze the layer:
    # If the layer has learnable parameters, they will not be updated.
    # When the layer is in the 'frozen' state (frozen = True), it will not be trained.
    def freeze(self):
        self.frozen = True

    # Cancel a previous freezing request: if the layer has learnable parameters, they will be updated again.
    def unfreeze(self):
        self.frozen = False

    # ---- YOUR TASK 2) ENDS HERE ----

    def get_parameters(self):
        pass  # Can be overwritten if necessary

    def set_parameters(self, parameters):
        pass  # Can be overwritten if necessary

    def persist(self, folder):
        pass  # Can be overwritten if necessary

    def load(self, folder):
        pass  # Can be overwritten if necessary

    def forward(self, input):
        raise NotImplementedError("Must be implemented by subclass.")

    def backprop(self, upstream_grad):
        raise NotImplementedError("Must be implemented by subclass.")

    def update_parameters(self, learning_rate):
        pass  # Can be overwritten if necessary
