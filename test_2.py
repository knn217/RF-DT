from evaluation import accuracy, balanced_accuracy, one_hot_encode, f1_score, precision, recall, train_test_split
from decision_tree import DecisionTree
from random_forest import RandomForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

###### OUR MODEL ######
def get_predictions_dt_ours(X_train, y_train, X_test, y_test):
    model = DecisionTree(min_samples_split=2, max_depth=2, metric='gini')
    model.fit(X_train, y_train)
    model.print_tree(columns = columns)
    predictions = model.predict(X_test)
    print("--- Our Model (DT) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")

def get_predictions_rf_ours(X_train, y_train, X_test, y_test):
    model = RandomForest(min_samples_split=2, max_depth=2, n_trees=10)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("--- Our Model (RF) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")


###### SKLEARN MODEL ######
def get_predictions_dt_sklearn(X_train, y_train, X_test, y_test):
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(X_train, y_train)
    predictions = decision_tree_classifier.predict(X_test)

    # Calculate evaluating metrics
    print("--- Sklearn's Model (DT) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")

def get_predictions_rf_sklearn(X_train, y_train, X_test, y_test):
    random_forest_classifier = RandomForestClassifier(min_samples_split=2, max_depth=2, n_estimators=10)
    random_forest_classifier.fit(X_train, y_train.ravel())
    predictions = random_forest_classifier.predict(X_test)

    # Calculate evaluating metrics
    print("--- Sklearn's Model (RF) ---")
    print(f"Model's Accuracy: {accuracy(y_test, predictions)}")

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
    df = pd.read_csv("drug200.csv")
    df_X = df.drop('Drug', axis=1)
    df_encoded = one_hot_encode(df_X)

    X = df_encoded.values
    y = df['Drug'].values.reshape(-1,1)

    columns = df_encoded.columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    get_predictions_dt_ours(X_train, y_train, X_test, y_test)
    get_predictions_dt_sklearn(X_train, y_train, X_test, y_test)

    get_predictions_rf_ours(X_train, y_train, X_test, y_test)
    get_predictions_rf_sklearn(X_train, y_train, X_test, y_test)
