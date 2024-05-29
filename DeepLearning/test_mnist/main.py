import numpy as np
from activation_functions import Softmax
from cost_functions import CrossEntropyCost
from persistence import persist, load
from training import train
from testing import forward
from dataset import download_mnist_dataset, download_cifar10_dataset, load_mnist_training_set, load_mnist_test_set, load_cifar10_dataset
from dataset import display_mnist_images
from display import print_test_progress, print_cost
from tools import SynchronousTimer, partition, make_batches
from model import Model
import time

#####################################################
# Configuration                                     #
#####################################################
# Folders:
folder_dataset_mnist = "./tmp/mnist"
folder_dataset_cifar10 = "./tmp/cifar10"
folder_model_load = "../transfer-learning/windows"
folder_model_store = "./model"

# Usage Parameters:
use_conv_as_fc = True
use_adam = True

# Flags:
flag_load_model = False
flag_autosave_model = True
flag_test = False
flag_download_mnist = False
flag_download_cifar10 = False

# Training Parameters:
m = 16
num_training_images = 6000
epochs = 100
learning_rate = 0.001 if use_adam else 0.1
initial_learning_rate = learning_rate
decay_rate = 0.01
decay_step = 0

# Checkpoint Parameters:
# Valid syntax:     "<1-59>s"
#                   "<1-59>m"
#                   "<1-24>h"
checkpoint_interval = "5m"

#####################################################
# Preparation of the Dataset                        #
#####################################################
# if flag_download_mnist:
#     download_mnist_dataset(folder_dataset_mnist)
# if flag_download_cifar10:
#     download_cifar10_dataset(folder_dataset_cifar10)

#####################################################
# Load the Training/Test sets                       #
#####################################################
X_train = None
Y_train = None
X_test = None
Y_test = None

if flag_test:
    m = 1
    X_test, Y_test = load_mnist_test_set(folder_dataset_mnist)

else:
    X_train, Y_train = load_mnist_training_set(folder_dataset_mnist)
    X_train = X_train[:num_training_images]
    Y_train = Y_train[:num_training_images]

# Using MNIST: input examples size = 1x28x28
shape_input_examples = (1, 28, 28)

np.random.seed(923893341)

#####################################################
# Definition of the Convolutional Model             #
#####################################################
model = Model()
model   .add("conv", "CONV1", num_filters=16, filter_size=3, stride=1, input_shape=shape_input_examples)\
        .add("pool", "POOL1", pool_size=2, stride=2)\
        .add("conv", "CONV2", num_filters=32, filter_size=3, stride=1)\
        .add("pool", "POOL2", pool_size=2, stride=2)

if use_conv_as_fc:
    model   .add("fc_conv", "FC1(conv)", num_filters=128, filter_size=7, stride=1, same_convolution=False, use_as_fc=True) \
            .add("fc_conv", "FC2(conv)", num_filters=10, filter_size=1, stride=1, same_convolution=False, use_as_fc=True)
else:
    model   .add("fc", "FC1", output_neurons=128) \
            .add("fc", "FC2", output_neurons=10)

model.add("output", "OUT1", activation_fn=Softmax(),
          cost_fn=CrossEntropyCost())
model.summary()

# Load the model if requested
if flag_load_model:
    load(model, folder_model_load)

#####################################################
# Testing                                           #
#####################################################
if flag_test:

    total = len(X_test)
    seen = 0
    correct = 0
    X_examples = partition(X_test, 1)
    Y_examples = partition(Y_test, 1)

    for i in range(len(X_examples)):
        x = X_examples[i]
        y = Y_examples[i]

    # Forward Pass
        y_pred = forward(model, x)

        # If the prediction corresponds to the label, increment the count of correct predictions
        if np.argmax(y) == np.argmax(y_pred):
            correct += 1
        seen += 1

        # Compute the instant accuracy
        instant_accuracy = float(correct)/float(seen)

        # Display the progress
        print_test_progress(instant_accuracy, i, total)

    # Perform inference to allow for visual verification
    for i in range(len(X_examples)):
        x = X_examples[i]

        # Forward Pass
        y_pred = forward(model, x)

    # Display the image with the predicted label
        display_mnist_images([x.reshape(28, 28)], [
                             np.argmax(y_pred)], caption="Prediction")

#####################################################
# Training                                          #
#####################################################
else:

    # Start measuring training time
    start_time = time.time()

    # If requested, define the frequency of Model Checkpoints:
    # The model can be persisted on a regular basis in order to avoid losing progress if the process is interrupted
    timer = SynchronousTimer()
    timer.add(checkpoint_interval, lambda: persist(
        model, folder_model_store), "Checkpoint: model persisted.")

    # Loop over the Training Epochs
    for epoch in range(epochs):

        # Shuffle the training set (maintains relative ordering between X and Y) and make batches
        X_batches, Y_batches, num_batches = make_batches(X_train, Y_train, m)

        # Loop over the batches
        for i in range(num_batches):

            # Extract the current batch:
            # X: batch of input examples, Y: batch of corresponding labels
            X = X_batches[i]
            Y = Y_batches[i]

    # Perform one Training step:
    # - Forward Pass of the input batch X
    # - Backward Pass of the gradient
    # - Update of model parameters
            cost = train(model, X, Y, learning_rate, use_adam)

    # Tick the timer to execute the scheduled actions
            timer.tick()

    # Print the current value of the cost function
            current_time = time.time()
            print_cost(start_time, current_time, cost,
                       epoch, epochs, i, num_batches)

        # If requested, persist the model at the end of every epoch
        if flag_autosave_model:
            persist(model, folder_model_store)
