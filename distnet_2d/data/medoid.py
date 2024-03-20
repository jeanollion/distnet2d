from numba import jit
import numpy as np
# Adapted from Original work Copyright 2021 Manan Lalit, Max Planck Institute of Molecular Cell Biology and Genetics  (MIT License https://github.com/juglab/EmbedSeg/blob/main/LICENSE)

def get_medoid(Y, X):
    dist_matrix = pairwise_python(Y, X)
    imin = np.argmin(np.sum(dist_matrix, axis=0))
    return Y[imin], X[imin]

@jit(nopython=True)
def pairwise_python(Y, X):
    N = Y.shape[0]
    D = np.zeros((N, N), dtype=np.float32)
    for i in range(0, N-1):
        for j in range(i+1, N):
            D[i, j] = np.sqrt( (Y[i] - Y[j])**2 + (X[i] - X[j])**2 )
            D[j, i] = D[i, j]
    return D