import numpy as np
from training import train

#############################################################################################################
# YOU HAVE TO DO THE FOLLOWING TASKS:                                                                       #
#                                                                                                           #
# TASK 1) Define the initial learning rate and the constant base for Exponential Decay                      #
# TASK 2) Update the learning rate at the beginning of each epoch based on the Exponential Decay formula    #
#############################################################################################################

# ---- YOUR TASK 1) STARTS HERE ----

alpha_0 = 0.2  # <YOUR CODE HERE> ####
base = .97  # <YOUR CODE HERE> ####

# ---- YOUR TASK 1) ENDS HERE ----

# Define how many training epochs you would like to perform
epochs = 20

# Loop over the training epochs
for i in range(epochs):

    # ---- YOUR TASK 2) STARTS HERE ----

    # Calculate the new learning rate using Exponential Decay
    alpha = alpha_0*base*(i/epochs)

    # ---- YOUR TASK 2) ENDS HERE ----

    # Execute one training step -- please do not modify the shapes as they are used in the test-case
    X = np.random.randn(5, 3, 28, 28)
    Y = np.random.randn(5, 10)
    train(X, Y, alpha)
    print(alpha)
