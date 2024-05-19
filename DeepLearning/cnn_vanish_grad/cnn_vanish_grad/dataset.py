import urllib.request
import pickle
import numpy as np
import gzip
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np

def download_mnist_dataset(folder):
    print(f"Downloading MNIST dataset into {folder}...")
    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

    if not os.path.exists(folder):
        os.makedirs(folder)

    for file in files:
        url = base_url + file
        destination = os.path.join(folder, file)
        urllib.request.urlretrieve(url, destination)
        print(f"Downloaded {file}")

    print("MNIST dataset downloaded successfully.")

def download_cifar10_dataset():
    data_folder = "cifar-10"
    print("Downloading CIFAR-10 dataset...")
    base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "cifar-10-python.tar.gz"

    if not os.path.exists("cifar-10"):
        os.makedirs("cifar-10")

    destination = os.path.join("cifar-10", file_name)
    urllib.request.urlretrieve(base_url, destination)

    tar = tarfile.open(destination, "r:gz")
    tar.extractall(path=data_folder)
    tar.close()

    print("CIFAR-10 dataset downloaded successfully.")

def load_mnist_dataset(folder):
    train_images = load_mnist_images(f"{folder}/train-images-idx3-ubyte.gz")
    train_labels = load_mnist_labels(f"{folder}/train-labels-idx1-ubyte.gz")

    test_images = load_mnist_images(f"{folder}/t10k-images-idx3-ubyte.gz")
    test_labels = load_mnist_labels(f"{folder}/t10k-labels-idx1-ubyte.gz")

    return train_images, train_labels, test_images, test_labels

def load_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 28, 28)
    return data

def load_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_cifar10_dataset():
    data_folder = "cifar-10/cifar-10-batches-py/"

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

    return train_images, train_labels, test_images, test_labels

def display_mnist_images(images, labels):
    fig, ax = plt.subplots()

    for i in range(len(images)):
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f"Label: {labels[i]}")
        plt.draw()
        plt.pause(1)
        ax.clear()

    plt.close()

def display_mnist_image(image, label, fig, ax):
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Label: {label}")
    plt.draw()
    plt.pause(2)
    ax.clear()

import matplotlib.pyplot as plt
import numpy as np

def display_cifar10_images(images, labels):
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    fig = plt.figure(figsize=(6, 6))

    for i in range(len(images)):
        image = images[i].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(f"Label: {class_names[labels[i]]}")
        plt.draw()
        plt.pause(1)
        if i < len(images) - 1:
            plt.clf()

    plt.close()
