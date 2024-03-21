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
    D = np.empty((N, N), dtype=np.float32) # np.empty -> faster than np.zeros
    for i in range(0, N-1):
        for j in range(i+1, N):
            D[j, i] = D[i, j] = np.sqrt( (Y[i] - Y[j])**2 + (X[i] - X[j])**2 )
    for i in range(N): # diagonal is always zero but must be filled as np.empty is used
        D[i, i] = 0.0
    return D