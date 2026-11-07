import numpy as np

class NaiveBayesModel:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + 1e-9
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def _calculate_likelihood(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def predict_proba(self, X):
        probs = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self._priors[idx])
                likelihood = np.sum(np.log(self._calculate_likelihood(idx, x) + 1e-9))
                posterior = prior + likelihood
                posteriors.append(posterior)
            exp_post = np.exp(posteriors - np.max(posteriors))
            prob_dist = exp_post / np.sum(exp_post)
            probs.append(prob_dist)
        return np.array(probs)

    def predict(self, X):
        prob = self.predict_proba(X)
        return np.argmax(prob, axis=1)
