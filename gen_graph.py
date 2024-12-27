import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please proved path to file with results as arg")
    results_path = sys.argv[1]
    results = pd.read_csv(results_path, header=None)
    print(results)
    plt.figure(figsize=(10, 6))
    plt.plot(results[0], results[1], marker='o')
    plt.xticks(results[0])
    plt.title('K-means clustering speedup')
    plt.xlabel('Number of threads per CUDA block')
    plt.ylabel('Speedup')
    plt.grid(True)
    plt.savefig('speedup.png')
    plt.close()
    

