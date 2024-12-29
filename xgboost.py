import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self, max_depth=3, min_samples_split=2, reg_lambda=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.reg_lambda = reg_lambda
        self.root = None
    
    def fit(self, X, grad, hess):
        self.root = self._build_tree(X, grad, hess, depth=0)

    def _build_tree(self, X, grad, hess, depth):
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return TreeNode(value=self._compute_leaf_value(grad, hess))

        best_feature, best_threshold, best_gain = None, None, -np.inf
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._compute_gain(X, grad, hess, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature, best_threshold = feature, threshold

        if best_gain == -np.inf:
            return TreeNode(value=self._compute_leaf_value(grad, hess))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left_child = self._build_tree(X[left_idx], grad[left_idx], hess[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx], grad[right_idx], hess[right_idx], depth + 1)

        return TreeNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)