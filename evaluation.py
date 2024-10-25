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

def balanced_accuracy(y_true, y_pred):
    """
    Calculates the balanced accuracy of a given array, for multi-class classification problems.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The balanced accuracy of the array.
    """
    y_pred = np.array(y_pred)
    y_true = y_true.flatten()
    # Get the number of classes
    n_classes = len(np.unique(y_true))

    # Initialize an array to store the sensitivity and specificity for each class
    sen = []
    spec = []
    # Loop over each class
    for i in range(n_classes):
        # Create a mask for the true and predicted values for class i
        mask_true = y_true == i
        mask_pred = y_pred == i

        # Calculate the true positive, true negative, false positive, and false negative values
        TP = np.sum(mask_true & mask_pred)
        TN = np.sum((mask_true != True) & (mask_pred != True))
        FP = np.sum((mask_true != True) & mask_pred)
        FN = np.sum(mask_true & (mask_pred != True))

        # Calculate the sensitivity (true positive rate) and specificity (true negative rate)
        if TP + FN == 0:
            sensitivity = 0
        else:
            sensitivity = TP / (TP + FN)
        if TN + FP == 0:
            specificity = 0
        else:
            specificity = TN / (TN + FP)

        # Store the sensitivity and specificity for class i
        sen.append(sensitivity)
        spec.append(specificity)
    # Calculate the balanced accuracy as the average of the sensitivity and specificity for each class
    average_sen =  np.mean(sen)
    average_spec =  np.mean(spec)
    balanced_acc = (average_sen + average_spec) / n_classes

    return balanced_acc

def classification_error(y_true, y_pred):
    """
    Calculates the classification error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The classification error of the array.
    """
    return 1 - accuracy(y_true, y_pred)

def mean_squared_error(y_true, y_pred):
    """
    Calculates the mean squared error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The mean squared error of the array.
    """
    return np.mean((y_true - y_pred)**2)

def mean_absolute_error(y_true, y_pred):
    """
    Calculates the mean absolute error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The mean absolute error of the array.
    """
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    """
    Calculates the R^2 score of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The R^2 score of the array.
    """
    mean = np.mean(y_true)
    ss_tot = np.sum((y_true - mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_res / ss_tot)

def mean_squared_log_error(y_true, y_pred):
    """
    Calculates the mean squared log error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The mean squared log error of the array.
    """
    return np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The mean absolute percentage error of the array.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    ndarray: The confusion matrix of the array.
    """
    unique_labels = np.unique(y_true)
    matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))
    return matrix

def precision(y_true, y_pred, average='macro'):
    """
    Calculates the precision of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.
    average (str): The averaging method to use. Default is 'macro'.

    Returns:
    float: The precision of the array.
    """
    matrix = confusion_matrix(y_true, y_pred)
    if average == 'macro':
        sums = np.sum(matrix, axis=0)
        sums[sums == 0] = 1  # avoid division by zero
        return np.mean(np.diag(matrix) / sums)
    elif average == 'micro':
        return np.sum(np.diag(matrix)) / np.sum(matrix)
    else:
        raise ValueError("Invalid averaging method. Use 'macro' or 'micro'")
    
def recall(y_true, y_pred, average='macro'):
    """
    Calculates the recall of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.
    average (str): The averaging method to use. Default is 'macro'.

    Returns:
    float: The recall of the array.
    """
    matrix = confusion_matrix(y_true, y_pred)
    if average == 'macro':
        return np.mean(np.diag(matrix) / np.sum(matrix, axis=1))
    elif average == 'micro':
        return np.sum(np.diag(matrix)) / np.sum(matrix)
    else:
        raise ValueError("Invalid averaging method. Use 'macro' or 'micro'")
    
def f1_score(y_true, y_pred, average='macro'):
    """
    Calculates the F1 score of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.
    average (str): The averaging method to use. Default is 'macro'.

    Returns:
    float: The F1 score of the array.
    """
    return 2 * (precision(y_true, y_pred, average) * recall(y_true, y_pred, average)) / (precision(y_true, y_pred, average) + recall(y_true, y_pred, average))

def accuracy_score(y_true, y_pred):
    """
    Calculates the accuracy score of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The accuracy score of the array.
    """
    return np.sum(y_true == y_pred) / len(y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates the mean absolute percentage error of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The mean absolute percentage error of the array.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

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