import numpy as np

class KMeansSequential:
    """
    A simple K-Means implementation that runs on the CPU using NumPy.

    Parameters
    ----------
    n_clusters : int, optional (default=5)
        The number of clusters (centroids) to form.
    max_iter : int, optional (default=1000)
        The maximum number of iterations to run.
    tol : float, optional (default=1e-3)
        The tolerance for convergence. If the change in centroids' position
        is less than this value, the algorithm will stop early.
    """
    def __init__(self, n_clusters=5, max_iter=1000, tol=1e-3):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """
        Perform K-Means clustering on the given data.

        Parameters
        ----------
        X : 2D array-like of shape (n_samples, n_features)
            The input data on which to perform clustering.

        Returns
        -------
        centroids : 2D array of shape (n_clusters, n_features)
            The final computed centroids.
        labels : 1D array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        # Randomly pick 'n_clusters' distinct points from X as initial centroids
        c_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[c_indices]

        # Main K-Means iteration
        for i in range(self.max_iter):
            # Print the current iteration (overwrites the same line)
            print(f"Iter {i+1}/{self.max_iter}\r", end="")

            # Compute the distance between each point in X and each centroid
            #   X[:, np.newaxis] shape: (n_samples, 1, n_features)
            #   centroids shape: (n_clusters, n_features)
            #   broadcasting => (n_samples, n_clusters, n_features)
            #   then use np.linalg.norm with axis=2 to get the Euclidean distance
            dists = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Find the index of the closest centroid for each point
            labels = np.argmin(dists, axis=1)

            # Recalculate the centroids based on current assignments
            new_centroids = np.array([
                X[labels == k].mean(axis=0)  # mean of all points assigned to cluster k
                for k in range(self.n_clusters)
            ])

            # Check for convergence:
            # if the magnitude of change in centroids < tol, stop
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break

            # Otherwise, update centroids for the next iteration
            centroids = new_centroids

        # Return the final centroids and labels
        return centroids, labels

