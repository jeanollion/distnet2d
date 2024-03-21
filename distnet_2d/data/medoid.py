from numba import jit
import numpy as np

def get_medoid(Y, X):
    dist_array = distarray_python(Y, X)
    imin = np.argmin(dist_array)
    return Y[imin], X[imin]

@jit(nopython=True)
def distarray_python(Y, X):
    N = Y.shape[0]
    D = np.zeros((N,), dtype=np.float32)
    for i in range(0, N-1):
        for j in range(i+1, N):
            d = np.sqrt( (Y[i] - Y[j])**2 + (X[i] - X[j])**2 )
            D[j] += d
            D[i] += d
    return D