class LayerBase:
    def __init__(self, name):
        self.name = name

    def forward(self, input):
        raise NotImplementedError("Must be implemented by subclass.")

    def backprop(self, upstream_grad):
        raise NotImplementedError("Must be implemented by subclass.")

    def update_parameters(self, learning_rate):
        pass  # Can be overwritten if necessary
