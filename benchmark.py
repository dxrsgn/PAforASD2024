import numpy as np
import time
from sklearn.datasets import make_blobs
from kmeans_sequential import KMeansSequential
from kmeans_parallel import KMeansCuda

SEED = 1111

if __name__ == "__main__":
    # Generate data
    X, _ = make_blobs(n_samples=70000, n_features=50, centers=5, random_state=SEED)
    print(X.shape)

    np.random.seed(SEED)
    seq_start = time.perf_counter()
    kmeans_seq = KMeansSequential()
    centroids, _ = kmeans_seq.fit(X)
    seq_end = time.perf_counter()
    print(f"Sequential KMeans execution time (s): {seq_end - seq_start}")

    np.random.seed(SEED)
    kmeans_parallel = KMeansCuda(n_threads=32)
    parallel_start = time.perf_counter()
    centroids, _ = kmeans_parallel.fit(X)
    parallel_end = time.perf_counter()
    print(f"Parallel KMeans execution time (s): {parallel_end - parallel_start}")

