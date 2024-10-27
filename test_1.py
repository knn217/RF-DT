from evaluation import accuracy, confusion_score, train_test_split
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import os
import cProfile


def getDir(file_name):
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    dir = os.path.join(path, file_name)
    #print(dir)
    return dir

def saveToTxt(data, name):
    dir = getDir(name)
    with open(dir, 'w', encoding='utf8') as f:
        for line in data:
            #print(line)
            f.write(str(line))
    return

###### OUR MODEL ######
def get_predictions_dt_ours(X_train, y_train, X_test, y_test, columns):
    model = DecisionTree(min_samples_split=2, max_depth=2, metric='gini')
    model.fit(X_train, y_train)
    model.print_tree(columns = columns)
    saveToTxt(model.log_tree(columns = columns), "log/breast_cancer.txt")
    predictions = model.predict(X_test)
    print("--- Our Model (DT) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")
    precision, recall, f1_score = confusion_score(y_test, predictions, "macro")
    print(f"Model's F1 (Macro): {f1_score}")
    print(f"Model's Precision (Macro): {precision}")
    print(f"Model's Recall (Macro): {recall}")
    precision, recall, f1_score = confusion_score(y_test, predictions, "micro")
    print(f"Model's F1 (Micro): {f1_score}")
    print(f"Model's Precision (Micro): {precision}")
    print(f"Model's Recall (Micro): {recall}")


###### SKLEARN MODEL ######
def get_predictions_dt_sklearn(X_train, y_train, X_test, y_test):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    predictions = decision_tree_classifier.predict(X_test)

    # Calculate evaluating metrics
    print("--- Sklearn's Model (DT) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")
    precision, recall, f1_score = confusion_score(y_test, predictions, "macro")
    print(f"Model's F1 (Macro): {f1_score}")
    print(f"Model's Precision (Macro): {precision}")
    print(f"Model's Recall (Macro): {recall}")
    precision, recall, f1_score = confusion_score(y_test, predictions, "micro")
    print(f"Model's F1 (Micro): {f1_score}")
    print(f"Model's Precision (Micro): {precision}")
    print(f"Model's Recall (Micro): {recall}")

def scale(X):
    """
    Standardizes the data in the array X.

    Parameters:
        X (numpy.ndarray): Features array of shape (n_samples, n_features).

    Returns:
        numpy.ndarray: The standardized features array.
    """
    # Calculate the mean and standard deviation of each feature
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    # Standardize the data
    X = (X - mean) / std

    return X

def all():
    df = pd.read_csv(getDir("breast-cancer.csv"))
    names = ['radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'radius_se',
    'perimeter_se',
    'area_se',
    'compactness_se',
    'concavity_se',
    'concave points_se',
    'radius_worst',
    'texture_worst',
    'perimeter_worst',
    'area_worst',
    'smoothness_worst',
    'compactness_worst',
    'concavity_worst',
    'concave points_worst',
    'symmetry_worst',
    'fractal_dimension_worst']

    X = df[names].values
    y = df['diagnosis'].values.reshape(-1,1)
    columns = df[names].columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    get_predictions_dt_ours(X_train, y_train, X_test, y_test, columns)
    get_predictions_dt_sklearn(X_train, y_train, X_test, y_test)

    get_predictions_rf_ours(X_train, y_train, X_test, y_test)
    get_predictions_rf_sklearn(X_train, y_train, X_test, y_test)
    return

if __name__ == "__main__":
    cProfile.run('all()')
