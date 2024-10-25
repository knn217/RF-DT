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
    available_sizes = [1e3, 1e4, 1e5, 5e3, 5e4, 5e5]
    datasets = [f"fake_drug_data_{int(size)}.csv" for size in available_sizes]  # Add your dataset filenames here
    log_file = f"./benchmark/benchmark_drug_{datetime.now()}.txt"

    with open(log_file, "w") as log:
        for dataset in datasets:
            df = pd.read_csv(dataset)
            df_X = df.drop('Drug', axis=1)
            df_encoded = one_hot_encode(df_X)

            X = df_encoded.values
            y = df['Drug'].values.reshape(-1,1)

            columns = df_encoded.columns.values

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
