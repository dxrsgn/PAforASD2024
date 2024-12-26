import numpy as np
import time
import sklearn
import sys
from sklearn.datasets import make_blobs
from kmeans_sequential import KMeansSequential
from kmeans_parallel import KMeansCuda

SEED = 1111

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify number of workers as argument")
        sys.exit(1)
    # Generate data
    X, _ = make_blobs(n_samples=70000, n_features=50, centers=5, center_box=(1,1), random_state=SEED)

    np.random.seed(SEED)
    kmeans_seq = KMeansSequential(max_iter=1000)
    t0 = time.perf_counter()
    centroids_seq, _ = kmeans_seq.fit(X)
    t1 = time.perf_counter()
    elapsed_seq = t1 - t0
    print(f"Sequential KMeans execution time (s): {elapsed_seq}")

    n_threads = int(sys.argv[1])
    np.random.seed(SEED)
    kmeans_parallel = KMeansCuda(n_threads=n_threads,max_iter=1000)
    t0 = time.perf_counter()
    centroids_para, _ = kmeans_parallel.fit(X)
    t1 = time.perf_counter()
    elapsed_parallel = t1 - t0
    print(f"Parallel KMeans execution time (s): {elapsed_parallel}")
    print(f"Correctnes check: {np.allclose(centroids_seq,centroids_para)}")
    print(f"Speedup (sequential_time/parallel_time): {elapsed_seq/elapsed_parallel}")
    #for n_threads in [4, 8, 16, 32]:
    #    np.random.seed(SEED)
    #    kmeans_parallel = KMeansCuda(n_threads=n_threads)
    #    t0 = time.perf_counter()
    #    centroids_para, _ = kmeans_parallel.fit(X)
    #    t1 = time.perf_counter()
    #    elapsed_parallel = t1 - t0
    #    print(f"Parallel KMeans with {n_threads}x{n_threads} threads per block execution time (s): {elapsed_parallel}")
    #    print(f"Correctnes check: {np.allclose(centroids_seq,centroids_para)}")

