import numpy as np
from numba import cuda, float64, int64, int32, void
from math import sqrt


@cuda.jit(void(float64[:, :], float64[:,:], float64[:, :], int64, int64, int64))
def centr_pts_dist_kernel(
    data,
    centroids,
    distances,
    num_points,
    num_features,
    num_centroids
):
    """
    Compute the squared distance between each point and each centroid.

    Parameters
    ----------
    data : 2D device array (num_points x num_features)
    centroids : 2D device array (num_centroids x num_features)
    distances : 2D device array (num_points x num_centroids)
        Output array to store the squared distances.
    num_points : int
    num_features : int
    num_centroids : int
    """
    pt_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    centr_idx = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    if pt_idx < num_points and centr_idx < num_centroids:
        dist = 0.0
        for feat in range(num_features):
            diff = data[pt_idx, feat] - centroids[centr_idx, feat]
            dist = dist + diff * diff
        distances[pt_idx, centr_idx] = sqrt(dist)


@cuda.jit(void(float64[:,:], int32[:], int64, int64))
def argmin_kernel(
    distances,
    labels,
    num_points,
    num_centroids
):
    """
    Find the index of the closest centroid for each point.

    Parameters
    ----------
    distances : 2D device array (num_points x num_centroids)
    labels : 1D device array (num_points,)
        Output array of centroid indices for each point.
    num_points : int
    num_centroids : int
    """
    pt_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if pt_idx < num_points:
        min_dist = float('inf')
        min_idx = -1

        # Find the centroid with the smallest distance to the current point
        for c_idx in range(num_centroids):
            dist = distances[pt_idx, c_idx]
            if dist < min_dist:
                min_dist = dist
                min_idx = c_idx

        labels[pt_idx] = min_idx


@cuda.jit(void(float64[:,:], int32[:], float64[:,:], float64[:,:], int32[:], int64, int64))
def centr_sum_kernel(
    data,
    labels,
    centroids,
    partial_sums,
    partial_counts,
    num_points,
    num_features
):
    """
    Accumulate the sums and counts needed to compute new centroids.

    Parameters
    ----------
    data : 2D device array (num_points x num_features)
    labels : 1D device array (num_points,)
    centroids : 2D device array (num_centroids x num_features)  # Not updated in-place
    partial_sums : 2D device array (num_centroids x num_features)
        Accumulator for summing all points belonging to each centroid.
    partial_counts : 1D device array (num_centroids,)
        Accumulator for counting how many points belong to each centroid.
    num_points : int
    num_features : int
    """
    pt_idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    if pt_idx < num_points:
        c_idx = labels[pt_idx]
        for feat in range(num_features):
            # Atomically add the feature value of the point to the partial sum
            cuda.atomic.add(partial_sums, (c_idx, feat), data[pt_idx, feat])
        # Atomically increment the count of points assigned to this centroid
        cuda.atomic.add(partial_counts, c_idx, 1)


class KMeansCuda:
    """
    A simple KMeans implementation using CUDA via Numba.

    Parameters
    ----------
    n_clusters : int, optional (default=5)
        The number of clusters to form.
    max_iter : int, optional (default=1000)
        Maximum number of iterations of the k-means algorithm.
    tol : float, optional (default=1e-3)
        Relative tolerance with regards to inertia to declare convergence.
    n_threads : int, optional (default=16)
        Number of threads to use per block when launching CUDA kernels.
    """
    def __init__(self, n_clusters=5, max_iter=1000, tol=1e-3, n_threads=16):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads

    def init_cuda(self, X, centroids):
        """
        Allocate and transfer data to the device.

        Parameters
        ----------
        X : 2D array of shape (num_points, num_features)
        centroids : 2D array of shape (n_clusters, num_features)
        """
        # Ensure floating-point precision
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        if centroids.dtype != np.float64:
            centroids = centroids.astype(np.float64)

        num_points = X.shape[0]
        num_features = X.shape[1]

        # Allocate device arrays
        self.dists_cu = cuda.device_array((num_points, self.n_clusters), dtype=np.float64)
        self.labels_cu = cuda.device_array((num_points), dtype=np.int32)
        #self.part_counts_cu = cuda.device_array((self.n_clusters), dtype=np.int32)
        #self.part_sums_cu = cuda.device_array((self.n_clusters, num_features), dtype=np.float64)

        # Transfer data to the device
        self.X_cu = cuda.to_device(X)
        self.centroids_cu = cuda.to_device(centroids)

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : 2D array of shape (num_points, num_features)

        Returns
        -------
        centroids : 2D array (n_clusters x num_features)
            Final computed centroids.
        labels : 1D array (num_points,)
            Index of the cluster each sample belongs to.
        """
        c_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[c_indices]

        # Initialize device memory with data and centroids
        self.init_cuda(X, centroids)

        # Define thread and grid dimensions
        threads_2d = (self.n_threads, self.n_threads)
        threads_1d = self.n_threads

        for i in range(self.max_iter):
            print(f"Iteration {i+1}/{self.max_iter}\r", end="")

            blocks_per_grid_x = (X.shape[0] + threads_2d[0] - 1) // threads_2d[0]
            blocks_per_grid_y = (self.n_clusters + threads_2d[1] - 1) // threads_2d[1]
            blocks_2d = (blocks_per_grid_x, blocks_per_grid_y)
            # Compute distances between points and centroids
            centr_pts_dist_kernel[blocks_2d, threads_2d](
                self.X_cu,
                self.centroids_cu,
                self.dists_cu,
                X.shape[0],
                X.shape[1],
                self.n_clusters
            )

            blocks_1d = (X.shape[0] + threads_1d - 1) // threads_1d
            # Assign each point to the closest centroid
            argmin_kernel[blocks_1d, threads_1d](
                self.dists_cu,
                self.labels_cu,
                X.shape[0],
                self.n_clusters
            )

            #print(self.labels_cu.copy_to_host())
            # Could be optimized probably (for example init these arrays on cuda once and only zeroe them in loop)
            self.part_counts_cu = cuda.to_device(np.zeros((self.n_clusters), dtype=np.int32))
            self.part_sums_cu = cuda.to_device(np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64))
            # Accumulate sums and counts for new centroid calculation
            centr_sum_kernel[blocks_1d, threads_1d](
                self.X_cu,
                self.labels_cu,
                self.centroids_cu,
                self.part_sums_cu,
                self.part_counts_cu,
                X.shape[0],
                X.shape[1]
            )
            cuda.synchronize()

            # Copy partial sums/counts back to cpu
            part_sums = self.part_sums_cu.copy_to_host()
            part_counts = self.part_counts_cu.copy_to_host()
            # Compute new centroids on the cpu
            new_centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
            for k in range(self.n_clusters):
                count = part_counts[k]
                if count > 0:
                    new_centroids[k] = part_sums[k] / count
            #print(new_centroids)
            # Check for convergence
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break

            centroids = new_centroids
            self.centroids_cu = cuda.to_device(centroids)

        return centroids, self.labels_cu.copy_to_host()

