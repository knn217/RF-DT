class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # Decision (Internal) nodes
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # Leaf nodes
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
    def is_decision_node(self):
        return self.value is None

    def __str__(self):
        if self.is_decision_node():
            return f"Decision Node: {self.feature} <= {self.threshold}"
        else:
            return f"Leaf Node: {self.value}"