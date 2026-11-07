import numpy as np

class KNNModel:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict_proba(self, X):
        probs = []
        for x in X:
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            prob_1 = np.sum(k_nearest_labels == 1) / self.k
            probs.append([1 - prob_1, prob_1])
        return np.array(probs)

    def predict(self, X):
        prob = self.predict_proba(X)[:, 1]
        return np.where(prob >= 0.5, 1, 0)
