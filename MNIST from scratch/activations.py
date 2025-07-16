import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(predicitons, targets_one_hot):
    """
    Args: prediction - numpy.ndarray (output of softmax)
    targets_one)hot - numpy.ndarray (true labels in onehot encode)
    Returns: float, average cross entropy loss over the batch
    """
    epsilon = 1e-10
    predicitons = np.clip(predicitons, epsilon, 1. - epsilon)
    loss = -np.sum(targets_one_hot * np.log(predicitons)) / predicitons.shape[0]
    return loss