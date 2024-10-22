import numpy as np

def accuracy(y_true, y_pred):
    """
    Calculates the accuracy of a given array.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The accuracy of the array.
    """
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def balanced_accuracy(y_true, y_pred):
    """
    Calculates the balanced accuracy of a given array, for multi-class classification problems.

    Parameters:
    y_true (ndarray): A 1D array of true labels.
    y_pred (ndarray): A 1D array of predicted labels.

    Returns:
    float: The balanced accuracy of the array.
    """
    C = len(np.unique(y_true))
    sum = 0.0
    for i in range(C):
        TP = np.sum((y_pred == i) & (y_true == i))
        TN = np.sum((y_pred != i) & (y_true != i))
        FP = np.sum((y_pred == i) & (y_true != i))
        FN = np.sum((y_pred != i) & (y_true == i))
        sum += (TP / (TP + FN + 1e-10) + TN / (TN + FP + 1e-10)) / (2 * C)
    return sum

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
        return np.mean(np.diag(matrix) / np.sum(matrix, axis=0))
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
