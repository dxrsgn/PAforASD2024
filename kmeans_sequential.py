import numpy as np
from math import sqrt

class KMeansSequential:
    def __init__(self, n_clusters=5, max_iter=1000, tol=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        c_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[c_indices]

        for i in range(self.max_iter):
            print(f"Iter {i+1}/{self.max_iter}\r", end="")

            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(dists, axis=1)
            #print(labels)

            new_centroids = np.array([
                X[labels == k].mean(axis=0)
                for k in range(self.n_clusters)
            ])
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break

            centroids = new_centroids

        return centroids, labels

