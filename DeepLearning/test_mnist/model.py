from conv_layer import ConvolutionalLayer
from pooling_layer import PoolingLayer
from fc_layer import FullyConnectedLayer
from output_layer import OutputLayer
class Model:
    def __init__(self):
        self.layers = []  # List to store model layers

    def add(self, layer_type, name, *args, **kwargs):
        # Check if input_shape is explicitly given and remove it from kwargs if it's the first layer
        if self.layers:
            # Automatically set input_shape from the output_shape of the last layer
            kwargs['input_shape'] = self.layers[-1].output_shape
        elif 'input_shape' in kwargs and not self.layers:
            # For the first layer, input_shape must be explicitly given
            pass
        else:
            raise ValueError("input_shape must be provided for the first layer")

        # Create the layer based on its type
        if layer_type == "conv":
            layer = ConvolutionalLayer(name, *args, **kwargs)
        elif layer_type == "pool":
            layer = PoolingLayer(name, *args, **kwargs)
        elif layer_type == "fc":
            layer = FullyConnectedLayer(name, *args, **kwargs)
        elif layer_type == "fc_conv":
            layer = ConvolutionalLayer(name, *args, **kwargs)  # Using a convolutional layer as a fully connected layer
        elif layer_type == "output":
            # Ensure that 'input_shape' is not passed to OutputLayer
            if 'input_shape' in kwargs:
                del kwargs['input_shape']
            layer = OutputLayer(name, activation_fn=kwargs.pop('activation_fn'), cost_fn=kwargs.pop('cost_fn'), *args, **kwargs)

        self.layers.append(layer)
        return self  # Allows for method chaining with .add calls

    def summary(self):
        # Intestazioni delle colonne per il sommario
        print(f"{'Layer Type':<20}{'Name':<15}{'Input Shape':<20}{'Output Shape':<20}{'Parameters':<30}")
        print('-' * 105)

        for layer in self.layers:
            # Estrai informazioni comuni a tutti i layer
            layer_type = layer.type
            name = layer.name
            input_shape = str(layer.input_shape) if hasattr(layer, 'input_shape') else 'dynamic'
            output_shape = str(layer.output_shape) if hasattr(layer, 'output_shape') else 'dynamic'

            # Estrai informazioni specifiche dei layer
            if isinstance(layer, ConvolutionalLayer):
                params = f"filters={layer.num_filters}, kernel_size={layer.filter_size}, stride={layer.stride}"
            elif isinstance(layer, PoolingLayer):
                params = f"pool_size={layer.pool_size}, stride={layer.stride}"
            elif isinstance(layer, FullyConnectedLayer):
                params = f"neurons={layer.output_neurons}" if hasattr(layer, 'output_neurons') else 'dynamic'
            elif isinstance(layer, OutputLayer):
                params = f"activation_fn={layer.activation_fn.__class__.__name__}, cost_fn={layer.cost_fn.__class__.__name__}"
            else:
                params = 'Custom Parameters'

            # Stampa le informazioni del layer
            print(f"{layer_type:<20}{name:<15}{input_shape:<20}{output_shape:<20}{params:<30}")
