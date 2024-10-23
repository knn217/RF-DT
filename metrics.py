import numpy as np

def gini(y):
    """
    Calculates the Gini index of a given array.

    Parameters:
    y (ndarray): A 1D array of labels.

    Returns:
    float: The Gini index of the array.
    """
    class_labels = np.unique(y)
    gini = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        gini += p_cls**2
    return 1 - gini

def entropy(y):
    """
    Calculates the entropy of a given array.

    Parameters:
    y (ndarray): A 1D array of labels.

    Returns:
    float: The entropy of the array.
    """

    class_labels = np.unique(y)
    entropy = 0
    for cls in class_labels:
        p_cls = len(y[y == cls]) / len(y)
        entropy += -p_cls * np.log2(p_cls)
    return entropy
