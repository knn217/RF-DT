# This contains code from Fares Elmenshawii.
# Link: https://www.kaggle.com/code/fareselmenshawii/decision-tree-from-scratch/notebook
# The code has been modified and extended by Sang Kha for research and educating purposes.

from metrics import entropy, gini
from node import Node
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, metric='gini'):
        """
        Initializes the DecisionTree with specified hyperparameters.

        Parameters:
        min_samples_split (int): The minimum number of samples required to split a node. Default is 2.
        max_depth (int): The maximum depth of the tree. Default is 100.
        n_feats (int or None): The number of features to consider when looking for the best split. 
                            If None, then all features are considered. Default is None.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.metric = metric
        self.root = None

    def fit(self, X, y):
        """
        Trains the DecisionTree model.

        Parameters:
        X (ndarray): A 2D array of features.
        y (ndarray): A 1D array of labels.

        Returns:
        DecisionTree: The trained model.
        """
        dataset = np.concatenate((X, y), axis=1)
        self.root = self._grow_tree(dataset)

    def predict(self, X):
        """
        Predicts the labels for the given set of features.

        Parameters:
        X (ndarray): A 2D array of features.

        Returns:
        ndarray: A 1D array of labels predicted by the model.
        """
        return np.array([self._make_prediction(x, self.root) for x in X])
    
    def _split(self, dataset, feature, threshold):
        """
        Splits the data based on the given threshold.

        Parameters:
        dataset (ndarray): A 2D array of features and labels.
        feature (int): Index of the feature to be split on.
        threshold (float): The threshold to split the data.

        Returns:
        left_dataset (ndarray): A 2D array of features and labels for the left dataset.
        right_dataset (ndarray): A 2D array of features and labels for the right dataset.
        """

        left_idxs = np.where(dataset[:, feature] <= threshold)[0]
        right_idxs = np.where(dataset[:, feature] > threshold)[0]

        left_dataset = dataset[left_idxs]
        right_dataset = dataset[right_idxs]

        return left_dataset, right_dataset
    
    
    def _split_categorical(self, dataset, feature, threshold):
        """
        Splits the data based on the given threshold for categorical variables.

        Parameters:
        dataset (ndarray): A 2D array of features and labels.
        feature (int): Index of the feature to be split on.
        threshold (set): A set of values to split the data.

        Returns:
        left_dataset (ndarray): A 2D array of features and labels for the left dataset.
        right_dataset (ndarray): A 2D array of features and labels for the right dataset.
        """
        left_idxs = np.where(np.isin(dataset[:, feature], threshold))[0]
        right_idxs = np.where(~np.isin(dataset[:, feature], threshold))[0]

        left_dataset = dataset[left_idxs]
        right_dataset = dataset[right_idxs]

        return left_dataset, right_dataset

    def _grow_tree(self, dataset, depth=0):     
        """
        Recursively grows the decision tree by selecting the best split at each node.

        Parameters:
        X (ndarray): A 2D array of features.
        y (ndarray): A 1D array of labels.
        depth (int): The current depth of the tree. Default is 0.

        Returns:
        Node: A node of the decision tree, which could be an internal decision node or a leaf node.
        """
        X, y = dataset[:,:-1], dataset[:,-1]
        n_samples, n_features = X.shape

        if n_samples >= self.min_samples_split and depth <= self.max_depth:
            best_split = self._best_criteria(dataset, n_features)
            if best_split["gain"] != 0:
                left = self._grow_tree(best_split["left_dataset"], depth+1)
                right = self._grow_tree(best_split["right_dataset"], depth+1)
                return Node(best_split["feature"], best_split["threshold"],
                            left, right, best_split["gain"])
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _best_criteria(self, dataset, num_features):
        """
        Finds the best feature and threshold to split the data.

        Parameters:
        dataset (ndarray): A 2D array of features and labels.
        num_features (int): The number of features in the dataset.

        Returns:
        dict: A dictionary with the best split feature index, threshold, gain, 
            left and right datasets.        
        """
        best_split = {
            "gain": -1,
            "feature": None,
            "threshold": None,
            "left_dataset": None,
            "right_dataset": None
        }
        for feat_idx in range(num_features):
            X_column = dataset[:, feat_idx]
            thresholds = np.unique(X_column)
            split_func = self._split if type(X_column[0]) is float else self._split_categorical
            for threshold in thresholds:
                left_dataset, right_dataset = split_func(dataset, feat_idx, threshold)
                if len(left_dataset) and len(right_dataset):
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    gain = self._information_gain(y, left_y, right_y, 'gini')
                    if gain > best_split["gain"]:
                        best_split["gain"] = gain
                        best_split["feature"] = feat_idx
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
        return best_split
    
    def _information_gain(self, parent, left, right, criterion='entropy'):        
        """
        Calculates the information gain of a split using the specified criterion.

        Parameters:
        y (ndarray): A 1D array of labels.
        X_column (ndarray): A 1D array of features.
        split_thresh (float): The threshold to split the data.
        criterion (str): The criterion to use for calculation ('entropy' or 'gini').

        Returns:
        float: The information gain of the split.
        """
        if criterion == 'entropy':
            measure = entropy
        elif criterion == 'gini':
            measure = gini
        else:
            raise ValueError("Criterion must be 'entropy' or 'gini'")

        parent_measure = measure(parent)

        left_measure, right_measure = measure(left), measure(right)
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        child_measure = weight_left * left_measure + weight_right * right_measure
        
        return parent_measure - child_measure
    
    def _make_prediction(self, x, node):
        """
        Traverses the decision tree to find the predicted label for a given feature vector.

        Parameters:
        x (ndarray): A 1D array of features.
        node (Node): The current node of the decision tree.

        Returns:
        int: The predicted label for the given feature vector.
        """
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._make_prediction(x, node.left)
        return self._make_prediction(x, node.right)

    def _most_common_label(self, Y):
        """
        Finds the most common label in a given array.

        Parameters:
        y (ndarray): A 1D array of labels.

        Returns:
        int: The most common label in the array.
        """
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self, columns=None, node=None, depth=0):
        """
        Prints the decision tree in a readable format.

        Parameters:
        node (Node): The current node of the decision tree.
        depth (int): The current depth of the tree. Default is 0.
        """
        if not node:
            node = self.root

        if node.is_leaf_node():
            print(f"{depth * '  '}Predict: {node.value}")
            return
        feature_name = node.feature if columns is None else columns[node.feature]
        feature_equality = "==" if type(node.threshold) is bool else "<="
        print(f"{depth * '  '}{feature_name} {feature_equality} {node.threshold}")

        self.print_tree(columns, node.left, depth + 1)
        self.print_tree(columns, node.right, depth + 1)
        
    def log_tree(self, columns=None, node=None, depth=0):
        """
        Prints the decision tree in a readable format.

        Parameters:
        node (Node): The current node of the decision tree.
        depth (int): The current depth of the tree. Default is 0.
        """
        if not node:
            node = self.root

        log = ""
        if node.is_leaf_node():
            log += f"{depth * '  '}Predict: {node.value}\n"
            return log
        feature_name = node.feature if columns is None else columns[node.feature]
        feature_equality = "==" if type(node.threshold) is bool else "<="
        log += f"{depth * '  '}{feature_name} {feature_equality} {node.threshold}\n"

        log += self.log_tree(columns, node.left, depth + 1)
        log += self.log_tree(columns, node.right, depth + 1)
        return log
