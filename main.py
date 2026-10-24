"""K-Means from Scratch - clustering with only NumPy, compared with scikit-learn."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


class KMeansScratch:
    def __init__(self, k, max_iter=100, seed=0):
        self.k = k
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

    def _init_centroids(self, X):
        """k-means++: spread the starting centroids out."""
        centroids = [X[self.rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None] - np.array(centroids)) ** 2).sum(axis=2), axis=1)
            centroids.append(X[self.rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centroids)

    def fit(self, X):
        self.centroids = self._init_centroids(X)
        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            self.labels = distances.argmin(axis=1)
            new = np.array([X[self.labels == c].mean(axis=0) if np.any(self.labels == c)
                            else self.centroids[c] for c in range(self.k)])
            if np.allclose(new, self.centroids):
                print(f"Converged after {i + 1} iterations")
                break
            self.centroids = new
        self.inertia = ((X - self.centroids[self.labels]) ** 2).sum()
        return self


def main():
    X, true_labels = make_blobs(n_samples=500, centers=4, cluster_std=0.9, random_state=3)

    mine = KMeansScratch(k=4).fit(X)
    sk = KMeans(n_clusters=4, n_init=10, random_state=0).fit(X)

    print(f"Scratch inertia:      {mine.inertia:.1f}")
    print(f"scikit-learn inertia: {sk.inertia_:.1f}")
    print(f"Agreement with true groups (ARI, 1.0 = perfect): {adjusted_rand_score(true_labels, mine.labels):.3f}")
    print("\nCentroids found:")
    for c in sorted(mine.centroids.tolist()):
        print(f"  ({c[0]:6.2f}, {c[1]:6.2f})")


if __name__ == "__main__":
    main()
