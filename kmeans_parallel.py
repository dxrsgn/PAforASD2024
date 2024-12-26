import numpy as np
from numba import cuda

@cuda.jit(cache=True)
def centr_pts_dist_kernel(
    X,
    centroids,
    dists,
    n_pts,
    n_feats,
    n_centroids
):
    pt_idx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    centr_idx = cuda.blockIdx.y*cuda.blockDim.y + cuda.threadIdx.y

    if pt_idx < n_pts and centr_idx < n_centroids:
        dist = 0.0
        for feat in range(n_feats):
            diff = (X[pt_idx, feat] - centroids[centr_idx, feat])
            dist += diff*diff
        dists[pt_idx, centr_idx] = dist
@cuda.jit(cache=True)
def argmin_kernel(
    dists,
    labels,
    n_pts,
    n_centroids
):
    pt_idx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if pt_idx < n_pts:
        min_dist = float('inf')
        min_idx = -1
        for centr_idx in range(n_centroids):
            dist = dists[pt_idx, centr_idx]
            if dist < min_dist:
                min_dist = dist
                min_idx = centr_idx
        labels[pt_idx] = min_idx
@cuda.jit(cache=True)
def centr_sum_kernel(
    X,
    labels,
    centroids,
    partial_sums,
    partial_counts,
    n_pts,
    n_features
):
    pt_idx = cuda.blockIdx.x*cuda.blockDim.x + cuda.threadIdx.x
    if pt_idx < n_pts:
        c = labels[pt_idx]
        for feat in range(n_features):
            cuda.atomic.add(partial_sums, (c, feat), X[pt_idx, feat])
        cuda.atomic.add(partial_counts, c, 1)
    
    
class KMeansCuda:
    def __init__(self, n_clusters=5, max_iter=1000, tol=1e-3, n_threads=16):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_threads = n_threads
    
    def init_cuda(self, X, centroids):
        if X.dtype != np.float64:
            X = X.astype(np.float64)
        if centroids.dtype != np.float64:
            centroids = centroids.astype(np.float64)

        n = X.shape[0]
        
        self.dists_cu = cuda.device_array((n, self.n_clusters), dtype=np.float64)
        self.labels_cu = cuda.device_array((n, ), dtype=np.int64)
        self.part_counts_cu = cuda.device_array((self.n_clusters, ), dtype=np.float64)
        self.part_sums_cu = cuda.device_array((self.n_clusters, X.shape[1]), dtype=np.float64)

        self.X_cu = cuda.to_device(X)
        self.centroids_cu = cuda.to_device(centroids)

    def fit(self, X):
        c_indicies = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        centroids = X[c_indicies]
        self.init_cuda(X, centroids)
        for i in range(self.max_iter):
            print(f"Iter {i+1}/{self.max_iter}\r", end = "")
            # Distance computation
            threads_per_block = (self.n_threads, self.n_threads)
            blocks_per_grid_x = (X.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
            blocks_per_grid_y = (self.n_clusters + threads_per_block[1] - 1) // threads_per_block[1]
            blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
            centr_pts_dist_kernel[blocks_per_grid, threads_per_block](
                self.X_cu,
                self.centroids_cu,
                self.dists_cu,
                X.shape[0],
                X.shape[1],
                self.n_clusters
            )
            # Label assingment
            threads_per_block1d = self.n_threads
            blocks_per_grid1d = (X.shape[0] + threads_per_block1d - 1) // threads_per_block1d
            argmin_kernel[blocks_per_grid1d, threads_per_block1d](
                self.dists_cu,
                self.labels_cu,
                X.shape[0],
                self.n_clusters
            )
            # New centroids calculation 
            threads_per_block1d = self.n_threads
            blocks_per_grid1d = (X.shape[0] + threads_per_block1d - 1) // threads_per_block1d 
            centr_sum_kernel[blocks_per_grid1d, threads_per_block1d](
                self.X_cu,
                self.labels_cu,
                self.centroids_cu,
                self.part_sums_cu,
                self.part_counts_cu,
                X.shape[0],
                X.shape[1]
            )
            part_sums = self.part_sums_cu.copy_to_host()
            part_counts = self.part_counts_cu.copy_to_host()
            new_centroids = np.zeros((self.n_clusters, X.shape[1]), dtype=np.float64)
            for k in range(self.n_clusters):
                count = part_counts[k]
                if count > 0:
                    new_centroids[k] = part_sums[k] / count

            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break
            centroids = new_centroids
            self.centroids_cu = cuda.to_device(centroids)

        return centroids, self.labels_cu.copy_to_host()


