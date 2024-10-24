from decision_tree import DecisionTree
import numpy as np

class RandomForest:
    def __init__(self, min_samples_split=2, n_trees=10, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Collect predictions from each tree
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return self._vote(tree_preds)

    def _vote(self, tree_preds):
        # Handle non-integer class labels by mapping them to integer values
        # First, find unique labels and map them to integers
        unique_labels, tree_preds_int = np.unique(tree_preds, return_inverse=True)
        tree_preds_int = tree_preds_int.reshape(tree_preds.shape)

        # Majority voting (mode) for each sample
        votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds_int)

        # Convert the votes back to the original labels
        return unique_labels[votes]


def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idx = np.random.choice(n_samples, n_samples, replace=True)
    return X[idx], y[idx]  # Ensure that the shape of y[idx] is 1D