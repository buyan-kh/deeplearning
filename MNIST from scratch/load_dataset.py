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
    """