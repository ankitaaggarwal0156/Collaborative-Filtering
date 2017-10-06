from __future__ import print_function

import sys
from collections import defaultdict
import numpy as np
from numpy import matrix
from pyspark import SparkContext
sc = SparkContext(appName="PythonALS")
LAMBDA = 0.01   # regularization
np.random.seed(42)

def ReadFile(filename,n,m):
    A=[]
    inp = defaultdict(lambda: defaultdict(int))
    mat1 = np.zeros((n,m))
    with open(filename, "r") as fo:
        for line in fo:
            A = line.split(",")
            mat1[int(A[0])-1][ int(A[1])-1]= int(A[2])
    fo.close()
    return mat1 

def rmse(R, ms, us):
    diff = R - ms * us.T
    return np.sqrt(np.sum(np.power(diff, 2)) / (M * U))


def update(i, vec, mat, ratings):
    uu = mat.shape[0]
    ff = mat.shape[1]

    XtX = mat.T * mat
    Xty = mat.T * ratings[i, :].T

    for j in range(ff):
        XtX[j, j] += LAMBDA * uu

    return np.linalg.solve(XtX, Xty)


if __name__ == "__main__":

    """
    Usage: als [M] [U] [F] [iterations] [partitions]"
    """

    print("""WARN: This is a naive implementation of ALS and is given as an
      example. Please use the ALS method found in pyspark.mllib.recommendation for more
      conventional use.""", file=sys.stderr)

    filename = "./"+sys.argv[1]
    M=int(sys.argv[2])
    U=int(sys.argv[3])
    F=int(sys.argv[4])
    ITERATIONS=int(sys.argv[5])
    partitions=int(sys.argv[6])
    outp = sys.argv[7]
    f = open(outp, "w")
    mat =ReadFile(filename,M,U)
    print("Running ALS with M=%d, U=%d, F=%d, iters=%d, partitions=%d\n" %(M, U, F, ITERATIONS, partitions))

    R = matrix(mat)
    ms = matrix(np.ones((M, F)))
    us = matrix(np.ones((U,F)))
    Rb = sc.broadcast(R)
    msb = sc.broadcast(ms)
    usb = sc.broadcast(us)

    for i in range(ITERATIONS):
        ms = sc.parallelize(range(M), partitions) \
               .map(lambda x: update(x, msb.value[x, :], usb.value, Rb.value)) \
               .collect()
        # collect() returns a list, so array ends up being
        # a 3-d array, we take the first 2 dims for the matrix
        ms = matrix(np.array(ms)[:, :, 0])
        msb = sc.broadcast(ms)

        us = sc.parallelize(range(U), partitions) \
               .map(lambda x: update(x, usb.value[x, :], msb.value, Rb.value.T)) \
               .collect()
        us = matrix(np.array(us)[:, :, 0])
        usb = sc.broadcast(us)

        error = rmse(R, ms, us)
        f.write("%5.4f\n" % error)
        
    f.close()
    sc.stop()
    
