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
    
    def _compute_gain(self, X, grad, hess, feature, threshold):
        left_idx = X[:, feature] <= threshold
        right_idx = X[:, feature] > threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return -np.inf

        grad_left, grad_right = grad[left_idx], grad[right_idx]
        hess_left, hess_right = hess[left_idx], hess[right_idx]

        gain = (
            (np.sum(grad_left)**2) / (np.sum(hess_left) + self.reg_lambda) +
            (np.sum(grad_right)**2) / (np.sum(hess_right) + self.reg_lambda) -
            (np.sum(grad)**2) / (np.sum(hess) + self.reg_lambda)
        )
        return gain
    
    def _compute_leaf_value(self, grad, hess):
        return -np.sum(grad) / (np.sum(hess) + self.reg_lambda)

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)