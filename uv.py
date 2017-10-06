import numpy as np
import sys
from collections import defaultdict

def ReadFile(filename,n,m):
    A=[]
    inp = defaultdict(lambda: defaultdict(int))
    mat = np.zeros((n,m))
    with open(filename, "r") as fo:
        for line in fo:
            A = line.split(",")
            mat[int(A[0])-1][ int(A[1])-1]= int(A[2])
    fo.close()
    return mat        
   

def UVdec(matrix,n,m,f,k, U, V):
    for r in xrange(n):
        for s in xrange(f):
            p=0
            a2=0
            for j in xrange(m):
                if(matrix[r][j]!=0):
                    p += V[s][j]*(matrix[r][j] - np.dot(U[r,:], V[:,j])+U[r][s]*V[s][j])
                    a2+= V[s][j]*V[s][j] 
            U[r][s]=p/a2
    for s in xrange(m):
        for r in xrange(f):
            p=0
            a2=0
            for i in xrange(n):
                if(matrix[i][s]!=0):
                    p += U[i][r]*(matrix[i][s] - np.dot(U[i,:], V[:,s])+U[i][r]*V[r][s])
                    a2+= U[i][r]*U[i][r] 
            V[r][s]=p/a2
    matN = np.matrix(np.dot(U, V))
    mD = np.array(matrix - matN)
    z = np.count_nonzero(matrix)
    sq_error =0
    for i in xrange(n):
        for j in xrange(m):
            if(matrix[i][j]!=0):
                sq_error = sq_error + mD[i][j]*mD[i][j]
    rmse = np.sqrt(sq_error/z)
    print "%.4f" %rmse
    
    return U,V

        
filename = "./"+sys.argv[1]
n = int(sys.argv[2])
m = int(sys.argv[3])
f = int(sys.argv[4])
k = int(sys.argv[5]) 

mat =ReadFile(filename,n,m)
U = np.ones((n,f))
V = np.ones((f,m))
U1,V1 = UVdec(mat,n,m,f,k, U, V)
for i in xrange(k-1):
    U1, V1 =  UVdec(mat,n,m,f,k, U1, V1)