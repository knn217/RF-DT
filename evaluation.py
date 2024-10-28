import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The accuracy of the array.
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    return (correct_predictions / total_samples) 

def confusion_score(y_true, y_pred, mode="macro"):
    """
    Calculates the F1 score for multi-label classification in either macro or micro mode.

    Parameters:
    y_true (np.ndarray): A 1D array of true labels.
    y_pred (np.ndarray): A 1D array of predicted labels.
    mode (str): Either 'macro' or 'micro'. Determines the F1 score calculation mode.

    Returns:
    float: The F1 score.
    """

    def one_hot_encode(arr: np.ndarray):
        # Get unique classes and their indices
        unique_classes, indices = np.unique(arr, return_inverse=True)
        
        # Create one-hot encoded matrix
        one_hot = np.zeros((arr.size, unique_classes.size))
        one_hot[np.arange(arr.size), indices] = 1
        
        return one_hot
    
    # One-hot encode the true and predicted labels
    y_true_one_hot = one_hot_encode(y_true)
    y_pred_one_hot = one_hot_encode(y_pred)
    
    # Adjust shapes if y_pred_one_hot has fewer columns
    if y_true_one_hot.shape[1] != y_pred_one_hot.shape[1]:
        y_pred_one_hot = np.pad(
            y_pred_one_hot,
            ((0, 0), (0, y_true_one_hot.shape[1] - y_pred_one_hot.shape[1])),
            mode="constant",
        )
    
    # True positives, predicted positives, actual positives per label
    true_positives = np.sum((y_true_one_hot == 1) & (y_pred_one_hot == 1), axis=0)
    predicted_positives = np.sum(y_pred_one_hot == 1, axis=0)
    actual_positives = np.sum(y_true_one_hot == 1, axis=0)
    
    # Calculate precision, recall, and F1 scores for each label
    precision_per_label = true_positives / (predicted_positives + 1e-9)
    recall_per_label = true_positives / (actual_positives + 1e-9)
    f1_per_label = 2 * (precision_per_label * recall_per_label) / (precision_per_label + recall_per_label + 1e-9)

    if mode == "macro":
        # Macro F1: Average F1 scores across labels
        f1_score = np.mean(f1_per_label)
        precision = np.mean(precision_per_label)
        recall = np.mean(recall_per_label)
        return precision, recall, f1_score
    elif mode == "micro":
        # Micro F1: Calculate global precision and recall and then F1
        total_true_positives = np.sum(true_positives)
        total_predicted_positives = np.sum(predicted_positives)
        total_actual_positives = np.sum(actual_positives)

        micro_precision = total_true_positives / (total_predicted_positives + 1e-9)
        micro_recall = total_true_positives / (total_actual_positives + 1e-9)
        f1_score = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-9)
        return micro_precision, micro_recall, f1_score
    else:
        raise ValueError("Mode should be 'macro' or 'micro'")


def train_test_split(X, y, random_state=40, test_size=0.2):
    """
    Splits the data into training and testing sets.

    Parameters:
    X (ndarray): A 2D array of features (n_samples, n_features).
    y (ndarray): A 1D array of labels (n_samples,).
    random_state (int): The random state to use. Default is 40.
    test_size (float): The proportion of the data to include in the test set. Default is 0.2.

    Returns:
    tuple: A tuple of (X_train, X_test, y_train, y_test).
    """
    idx = np.arange(len(X))
    np.random.seed(random_state)
    np.random.shuffle(idx)
    idx_test = idx[:int(test_size * len(X))]
    idx_train = idx[int(test_size * len(X)):]
    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    return X_train, X_test, y_train, y_test

def one_hot_encode(df: pd.DataFrame):
    """
    Performs custom one-hot encoding on categorical columns of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with columns to be encoded.

    Returns:
    pd.DataFrame: A new DataFrame with categorical columns one-hot encoded.
                    Numeric columns are included as-is.
    """
    one_hot_encoded = pd.DataFrame()
    for col in df.columns:
        if df[col].dtype == 'object':
            dummies = pd.get_dummies(df[col], prefix=col)
            one_hot_encoded = pd.concat([one_hot_encoded, dummies], axis=1)
        else:
            one_hot_encoded = pd.concat([one_hot_encoded, df[col]], axis=1)
    return one_hot_encoded