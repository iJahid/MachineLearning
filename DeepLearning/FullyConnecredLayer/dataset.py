import urllib.request
import pickle
import numpy as np
import gzip
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np

# Download the MNIST dataset
#
# Parameters:
# Input	folder:		Destination folder where the MNIST archive will be downloaded
def download_mnist_dataset(folder):
    print(f"Downloading MNIST dataset into {folder}...")

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

	# Create the folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    for file in files:
        url = base_url + file
        destination = os.path.join(folder, file)
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {file}")

    print("MNIST dataset downloaded successfully.")

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

# Convert class index to one-hot vector
#
# Parameters:
# Input		y:				Class index as a numeric integer value
# Input		num_classes:	Number of output classes in the dataset
# Output					One-hot label vector
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

# Load the MNIST training set
#
# Parameters:
# Input		folder:			Folder where the dataset gz archives are stored
# Output	mnist_data:		Array containing the normalized images
# Output	mnist_labels:	Array containing the one-hot label vectors
def load_mnist_training_set(folder):
    factor = 0.99 / 255
    filename = f"{folder}/train-images-idx3-ubyte.gz"
    mnist_data = load_mnist_images(filename)
    mnist_data = mnist_data * factor + 0.01  # Normalize the data

    filename = f"{folder}/train-labels-idx1-ubyte.gz"
    mnist_labels = load_mnist_labels(filename)
    mnist_labels = to_categorical(mnist_labels, 10)  # Assuming there are 10 classes in MNIST

    return mnist_data, mnist_labels

# Load the MNIST test set
#
# Parameters:
# Input		folder:			Folder where the dataset gz archives are stored
# Output	mnist_data:		Array containing the normalized images
# Output	mnist_labels:	Array containing the one-hot label vector
def load_mnist_test_set(folder):
    factor = 0.99 / 255
    filename = f"{folder}/t10k-images-idx3-ubyte.gz"
    mnist_data = load_mnist_images(filename)
    mnist_data = mnist_data * factor + 0.01  # Normalize the data

    filename = f"{folder}/t10k-labels-idx1-ubyte.gz"
    mnist_labels = load_mnist_labels(filename)
    mnist_labels = to_categorical(mnist_labels, 10)  # Assuming there are 10 classes in MNIST

    return mnist_data, mnist_labels

# Display the labelled MNIST images
#
# Parameters:
# Input	images:		List containing the 28x28x1 images from the dataset
# Input	labels:		List containing the corresponding labels / predictions
# Input caption:	Optional caption to be prepended to the object class instead of "Label"
def display_mnist_images(images, labels, caption="Label"):
    fig, ax = plt.subplots()

    for i in range(len(images)):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"{caption}: {labels[i]}")
        plt.draw()
        plt.pause(1)
        ax.clear()

    plt.close()

# Download the CIFAR-10 dataset
#
# Parameters:
# Input	folder:		Destination folder where the CIFAR-10 archive will be downloaded
def download_cifar10_dataset(folder):
    print(f"Downloading CIFAR-10 dataset into {folder}...")
    base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "cifar-10-python.tar.gz"

    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, file_name)
    urllib.request.urlretrieve(base_url, destination)

    tar = tarfile.open(destination, "r:gz")
    tar.extractall(path=folder)
    tar.close()

    print("CIFAR-10 dataset downloaded successfully.")

# Load the CIFAR-10 dataset
#
# Parameters:
# Input	folder:		Source folder from where to load the CIFAR-10 dataset
def load_cifar10_dataset(folder):
	data_folder = f"{folder}/cifar-10-batches-py/"
	factor = 0.99 / 255

	train_images, train_labels = [], []
	for batch_id in range(1, 6):
		file_path = f"{data_folder}/data_batch_{batch_id}"
		with open(file_path, 'rb') as file:
			batch = pickle.load(file, encoding='bytes')
			train_images.append(batch[b'data'])
			train_labels.append(batch[b'labels'])
	train_images = np.concatenate(train_images, axis=0)
	train_labels = np.concatenate(train_labels, axis=0)

	file_path = f"{data_folder}/test_batch"
	with open(file_path, 'rb') as file:
		batch = pickle.load(file, encoding='bytes')
		test_images = batch[b'data']
		test_labels = np.array(batch[b'labels'])
		train_labels = to_categorical(train_labels, 10)
		test_labels = to_categorical(test_labels, 10)

		train_images = train_images.astype('float32')
		test_images = test_images.astype('float32')

		train_images = train_images * factor + 0.01
		test_images = test_images * factor + 0.01

	return train_images.reshape(-1, 3, 32, 32), train_labels, test_images.reshape(-1, 3, 32, 32), test_labels

# Display the labelled CIFAR-10 images
#
# Parameters:
# Input images:     List containing the 32x32x3 images from the dataset
# Input labels:     List containing the corresponding labels / predictions
# Input caption:    Optional caption to be prepended to the object class instead of "Label"
def display_cifar10_images(images, labels, caption="Label"):
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    fig = plt.figure(figsize=(6, 6))

    for i in range(len(images)):
        image = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(f"{caption}: {class_names[labels[i]]}")
        plt.draw()
        plt.pause(1)
        if i < len(images) - 1:
            plt.clf()

    plt.close()
