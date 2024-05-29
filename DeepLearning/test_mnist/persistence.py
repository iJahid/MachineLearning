import os
import pickle

# Write the parameters of one layer to a binary file located in the specified folder
#
# Parameters:
# Input     parameters:     Dictionary containing the name and all parameters of the layer.
# Input     folder:         Destination folder where the parameters file must be stored.
def write_parameters(parameters, folder):
    # Ensure that the folder exists, otherwise, create it
    os.makedirs(folder, exist_ok=True)

    # Extract the layer name from the parameters dictionary
    layer_name = parameters["layer_name"]

    # Define the name of the binary file containing the parameters:
    # The filename must be the layer's name followed by the '.bin' extension.
    file_name = os.path.join(folder, f"{layer_name}.bin")

    # Write the parameters to the binary file
    # Hint: use the 'pickle.dump' function.
    with open(file_name, "wb") as file:
        pickle.dump(parameters, file)

# Read the parameters of one layer from a binary file located in the specified folder
#
# Parameters:
# Input     folder:         Source folder where the parameters file is stored.
# Input     layer_name:     Name of the layer.
# Output    parameters:     Dictionary containing the name and all parameters of the layer. 
def read_parameters(folder, layer_name):
    # Define the name of the binary file containing the parameters:
    # The filename must be the layer's name followed by the '.bin' extension.
    file_name = os.path.join(folder, f"{layer_name}.bin")

    # Verify that the file exists
    if not os.path.exists(file_name):
        raise ValueError(f"Parameters file '{file_name}' not found in folder '{folder}'.")

    # Read the parameters from the binary file
    # Hint: use the 'pickle.load' function.
    with open(file_name, "rb") as file:
        parameters = pickle.load(file)

    return parameters

# Persist the parameters of the network to the specified folder
#
# Parameters:
# Input     model:          Model containing the list of all layers of the network.
# Input     folder:         Destination folder where the parameters are persisted.
def persist(model, folder):
    for layer in model.layers:
        layer.persist(folder)
    
# Load the parameters of the network from the specified folder
#
# Parameters:
# Input     model:          Model containing the list of all layers of the network.
# Input     folder:         Source folder where the parameters are stored.
def load(model, folder):
	for layer in model.layers:
		layer.load(folder)
	print(f"Model loaded from {folder}")
