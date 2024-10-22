from metrics import entropy, gini
from node import Node
import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
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
        self.n_feats = n_feats
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
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Predicts the labels for the given set of features.

        Parameters:
        X (ndarray): A 2D array of features.

        Returns:
        ndarray: A 1D array of labels predicted by the model.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _split(self, X_column, split_thresh):
        """
        Splits the data based on the given threshold.

        Parameters:
        X_column (ndarray): A 1D array of features.
        split_thresh (float): The threshold to split the data.

        Returns:
        tuple: A tuple of the indices of the left and right data.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _grow_tree(self, X, y, depth=0):     
        """
        Recursively grows the decision tree by selecting the best split at each node.

        Parameters:
        X (ndarray): A 2D array of features.
        y (ndarray): A 1D array of labels.
        depth (int): The current depth of the tree. Default is 0.

        Returns:
        Node: A node of the decision tree, which could be an internal decision node or a leaf node.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # Greedy search
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Finds the best feature and threshold to split the data.

        Parameters:
        X (ndarray): A 2D array of features.
        y (ndarray): A 1D array of labels.
        feat_idxs (ndarray): A 1D array of feature indices to consider.

        Returns:
        tuple: A tuple of the best feature index and threshold.
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh, criterion='entropy'):        
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

        parent_measure = measure(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        measure_l, measure_r = measure(y[left_idxs]), measure(y[right_idxs])
        child_measure = (n_l / n) * measure_l + (n_r / n) * measure_r
        return parent_measure - child_measure
    
    def _traverse_tree(self, x, node):
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
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        """
        Finds the most common label in a given array.

        Parameters:
        y (ndarray): A 1D array of labels.

        Returns:
        int: The most common label in the array.
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
