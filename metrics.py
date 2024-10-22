import numpy as np

def gini(y):
    """
    Calculates the Gini index of a given array.

    Parameters:
    y (ndarray): A 1D array of labels.

    Returns:
    float: The Gini index of the array.
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    return 1 - np.sum(ps**2)

def entropy(y):
    """
    Calculates the entropy of a given array.

    Parameters:
    y (ndarray): A 1D array of labels.

    Returns:
    float: The entropy of the array.
    """
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
