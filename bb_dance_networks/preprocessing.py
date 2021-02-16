import numpy as np


class FeatureNormalizer:
    def __init__(self):
        self.percentiles = None

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

    def to_dict(self):
        return dict(
            percentiles=self.percentiles,
            data_means=self.data_means,
            data_stds=self.data_stds,
        )

    @staticmethod
    def from_dict(d):
        normalizer = FeatureNormalizer()
        normalizer.percentiles = d["percentiles"]
        normalizer.data_means = d["data_means"]
        normalizer.data_stds = d["data_stds"]
        return normalizer

    def save(self, filename):
        import joblib

        joblib.dump(self.to_dict(), filename)

    @staticmethod
    def load(filename):
        import joblib

        d = joblib.load(filename)
        return FeatureNormalizer.from_dict(d)
