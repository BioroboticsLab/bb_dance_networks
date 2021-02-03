import numpy as np


class FeatureNormalizer:

    percentiles = None

    def __init__(self):
        pass

    def fit(self, X, verbose=False):
        X = X.copy()
        self.percentiles = []
        for i in range(X.shape[1]):
            low, high = np.percentile(X[:, i].flatten(), (0.5, 99.5))
            self.percentiles.append((low, high))
            if verbose:
                print("Low: {}, High: {}".format(low, high))
            X[:, i, :] = np.clip(X[:, i, :], low, high)

        self.data_means = X.mean(axis=(0, 2))
        self.data_stds = X.std(axis=(0, 2))
        return self

    def transform(self, X):
        X = X.copy()
        for i in range(X.shape[1]):
            X[:, i, :] = np.clip(X[:, i, :], *self.percentiles[i])
            X[:, i] -= self.data_means[i]
            X[:, i] /= self.data_stds[i]
        return X
