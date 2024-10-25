from evaluation import accuracy, balanced_accuracy, one_hot_encode, f1_score, precision, recall, train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from datetime import datetime

###### OUR MODEL ######
import time

def get_predictions_dt_ours(X_train, y_train, X_test, y_test):
    start = time.time()
    model = DecisionTree(min_samples_split=2, max_depth=2, metric='gini')
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    end = time.time()
    return f"Model's Accuracy: {accuracy(y_test, predictions)}\nTime elapsed: {end - start} seconds\n"

def get_predictions_rf_ours(X_train, y_train, X_test, y_test):
    start = time.time()
    model = RandomForest(min_samples_split=2, max_depth=2, n_trees=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    end = time.time()
    return f"Model's Accuracy: {accuracy(y_test, predictions)}\nTime elapsed: {end - start} seconds\n"


###### SKLEARN MODEL ######
def get_predictions_dt_sklearn(X_train, y_train, X_test, y_test):
    start = time.time()
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    predictions = decision_tree_classifier.predict(X_test)
    end = time.time()
    # Calculate evaluating metrics
    return f"Model's Accuracy: {accuracy(y_test, predictions)}\nTime elapsed: {end - start} seconds\n"

def get_predictions_rf_sklearn(X_train, y_train, X_test, y_test):
    start = time.time()
    random_forest_classifier = RandomForestClassifier(min_samples_split=2, max_depth=2, n_estimators=10)
    random_forest_classifier.fit(X_train, y_train.ravel())
    predictions = random_forest_classifier.predict(X_test)
    end = time.time()
    # Calculate evaluating metrics
    return f"Model's Accuracy: {accuracy(y_test, predictions)}\nTime elapsed: {end - start} seconds\n"

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

if __name__ == "__main__":
    available_sizes = [1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4]
    datasets = [f"fake_breast_cancer_data_{int(size)}.csv" for size in available_sizes]  # Add your dataset filenames here
    log_file = f"./benchmark/benchmark_breast_cancer_{datetime.now()}.txt"

    with open(log_file, "w") as log:
        for dataset in datasets:
            df = pd.read_csv(dataset)
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

            log.write(f"--- Dataset: {dataset} ---\n")
            
            log.write("--- Our Model (DT) ---\n")
            log.write(get_predictions_dt_ours(X_train, y_train, X_test, y_test))

            log.write("--- Sklearn's Model (DT) ---\n")
            log.write(get_predictions_dt_sklearn(X_train, y_train, X_test, y_test))

            # log.write("--- Our Model (RF) ---\n")
            # get_predictions_rf_ours(X_train, y_train, X_test, y_test)

            # log.write("--- Sklearn's Model (RF) ---\n")
            # get_predictions_rf_sklearn(X_train, y_train, X_test, y_test)

            log.write("\n")
            print(f"--- Finished Dataset: {dataset} ---\n")
