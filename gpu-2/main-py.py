import math
import sys
import numpy as np
from numba import jit
from time import time
from matplotlib import pyplot as plt


@jit
def foo(x, y):
    return math.sin(x) * math.cos(y)

@jit
def calc_integral(N):
    x = np.array([i for i in np.arange(0, 1, 1/N)])
    y = np.array([i for i in np.arange(0, 1, 1/N)])
    integral = 0
    for i in range(x.shape[0]-1): # -> if global_id(0) < N
        for j in range(y.shape[0]-1): # -> for ... -> sums[global_id(0)]
            integral += foo((x[i]+x[i+1])/2, (y[j]+y[j+1])/2) * (x[i+1]-x[i]) * (y[j+1]-y[j])
    return integral

if __name__ == "__main__":
    argv = sys.argv[1:]
    try:
        N = int(argv[0])
        t0 = time()
        res = calc_integral(N)
        t1 = time()
        print("Integral:", res)
        print("Time:", t1-t0, "sec")
        print("Diff:", abs(0.3868223-res))
    except:
        mode = str(argv[0])
        if mode == "graph":
            for N in range(500, 10000, 100):
                t0 = time()
                res = calc_integral(N)
                t1 = time()
                delta = abs(0.3868223-res)
                plt.scatter(N, delta)
            plt.show()
