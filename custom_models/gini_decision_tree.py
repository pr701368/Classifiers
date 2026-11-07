import numpy as np

class GiniDecisionTreeModel:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def _gini(self, y):
        m = len(y)
        if m == 0: return 0
        p1 = np.sum(y == 1) / m
        p0 = 1 - p1
        return 1 - (p0**2 + p1**2)

    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1: return None, None
        
        best_gini = 999.0
        best_idx, best_thr = None, None
        
        for idx in range(n):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left_mask = X[:, idx] <= thr
                y_left, y_right = y[left_mask], y[~left_mask]
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                gini = (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / m
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = thr
                    
        return best_idx, best_thr

    def _build_tree(self, X, y, depth=0):
        num_labels = len(np.unique(y))
        if depth >= self.max_depth or num_labels == 1 or len(y) < 2:
            return np.round(np.mean(y))
        
        idx, thr = self._best_split(X, y)
        if idx is None:
            return np.round(np.mean(y))
            
        left_mask = X[:, idx] <= thr
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return {"feature_idx": idx, "threshold": thr, "left": left_node, "right": right_node}

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _predict_row(self, node, x):
        if not isinstance(node, dict):
            return node
        if x[node["feature_idx"]] <= node["threshold"]:
            return self._predict_row(node["left"], x)
        return self._predict_row(node["right"], x)

    def predict(self, X):
        return np.array([self._predict_row(self.tree, x) for x in X])
