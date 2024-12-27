# K-means Clustering with CUDA Parallelization


## Algorithm and Parallelization Method

In this work K-means was used as a subject for parallelization.  
Sequential K-means was implemented from scratch with numpy.  
Parallized version of K-means was implented with CUDA and numba library. Specifically, following parts were parallelized:  
- Calculation of distance between data and centroids. Custom CUDA 2d kernel was implemented to solve this task. First dimension of kernel was assigned to data points and second to cluster, thus in parallel by CUDA were calculated distances between each data point and each centroids  
- Label assingment. Custom 1d kernel was implemented to find argmin on distances vector.  
- Centroids summation. Custom 1d kernel was implemented to find sum of data points assigned to specific kernel. Parallelization was made on data points dimension with use of atomic operations. Thus, in parallel each data point added to global vector of centroids its values. To prevent racing, atomic operations were used. This part could be further improved with shared arrays and block summation techniques, since using atomic operations slows down computations because of blocking clusters.   

### How to run:
1. Clone the repository:
```
git clone https://github.com/dxrsgn/PAforASD2024.git
cd PAforASD2024
```
2. Install cuda toolkit  
3. Set up enviroment:  
```
pip install -r requiremnts.txt
```
4. Run the benchmark: 
```
./experiment.sh
```

## Data
Data is automatically generated on behchmark run using sklearn.datasets.make_blobs with specified random seed.  
Totaly there are 70000 samples, 50 features and 5 clusters.

## Speedup graph

In the experiment number of threads means number of threads per CUDA block, i.e. if kernel is 1d (argmin, cluster sum) there are executed totaly n_blocksXn_threads threads (number of blocks is automatically calculated to fit all the data) and in case of 2d kernel there are executed totaly n_blocks_xXn_threads + n_blocks_yXn_threads threads.  
In the plot belowe number of threads mean n_threads value.  




