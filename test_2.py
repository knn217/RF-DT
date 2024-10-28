from evaluation import accuracy, one_hot_encode, confusion_score, train_test_split
from decision_tree import DecisionTree
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import numpy as np
from test_1 import getDir, saveToTxt
import cProfile

###### OUR MODEL ######
def get_predictions_dt_ours(X_train, y_train, X_test, y_test, columns):
    model = DecisionTree(min_samples_split=2, max_depth=2, metric='gini')
    model.fit(X_train, y_train)
    model.print_tree(columns = columns)
    saveToTxt(model.log_tree(columns = columns), "log/drug.txt")
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

    print(export_text(decision_tree_classifier))

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
    df = pd.read_csv(getDir("drug200.csv"))
    df_X = df.drop('Drug', axis=1)
    df_encoded = one_hot_encode(df_X)

    X = df_encoded.values
    y = df['Drug'].values.reshape(-1,1)

    columns = df_encoded.columns.values

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    get_predictions_dt_ours(X_train, y_train, X_test, y_test, columns)
    get_predictions_dt_sklearn(X_train, y_train, X_test, y_test)
    return

if __name__ == "__main__":
    cProfile.run('all()')
