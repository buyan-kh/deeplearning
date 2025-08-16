import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt
import os

def load_mnist_images(filepath):
    """
    Args: filepath (str)
    Returns: numpy ndarray, 2d array of images
    """
    with gzip.open(filepath, 'rb') as f:
        magic, num_images, num_rows, num_cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic num")
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, num_rows, num_cols)
        return images / 255.0

def load_mnist_labels(filepath):
    """
    Args: filepath (str) : Path to the .gz label file
    Returns: numpy.ndarray: a 1D array of labels
    """
    with gzip.open(filepath, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"invalid magic numb")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_mnist_dataset(data_path='.'):
    """
    Args: directory where mnist .gz files are located
    Returns: tuple: (train_images, train_labels, test_images, test_labels)
        Images are flattened (num_samples, 784)
        Labels are 1D arrays
    """
    print("loading MNIST dataset...")
    train_images_path = os.path.join(data_path, './dataset/train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_path, './dataset/train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_path, './dataset/t10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_path, './dataset/t10k-labels-idx1-ubyte.gz')

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print("MNIST dataset loaded successfully!")

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist_dataset()
    fig, axes = plt.subplots(1, 5, figsize = (10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X_train[i].reshape(28, 28), cmap='gray')
        ax.set_title(f"label: {y_train[i]}")
        ax.axis('off')
    plt.suptitle("sample mnist training images")
    plt.show()

    print(f"unique training labels: {np.unique(y_train)}")
