from numba import jit
import numpy as np
# Original work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)

def get_medoid(Y, X):
    dist_matrix = pairwise_python(np.vstack((X, Y)).transpose())
    imin = np.argmin(np.sum(dist_matrix, axis=0))
    return Y[imin], X[imin]

@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D
