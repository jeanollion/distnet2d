from numba import jit
import numpy as np

def get_medoid(*coords):
    """Get medoid from coordinate arrays. Works for 2D, 3D or any number of dimensions.
    Args:
        *coords: variable number of 1D arrays (e.g. Y, X or Z, Y, X)
    Returns:
        tuple of coordinates of the medoid
    """
    ndim = len(coords)
    if ndim == 2:
        Y, X = coords
        dist_array = _distarray_2d(np.asarray(Y, dtype=np.float64), np.asarray(X, dtype=np.float64))
    elif ndim == 3:
        Z, Y, X = coords
        dist_array = _distarray_3d(np.asarray(Z, dtype=np.float64), np.asarray(Y, dtype=np.float64), np.asarray(X, dtype=np.float64))
    else:
        points = np.stack([np.asarray(c) for c in coords], axis=1).astype(np.float64)
        dist_array = _distarray_nd(points)
    imin = np.argmin(dist_array)
    return tuple(np.asarray(c)[imin] for c in coords)

@jit(nopython=True)
def _distarray_2d(Y, X):
    N = Y.shape[0]
    D = np.zeros((N,), dtype=np.float64)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            d = np.sqrt((Y[i] - Y[j])**2 + (X[i] - X[j])**2)
            D[j] += d
            D[i] += d
    return D

@jit(nopython=True)
def _distarray_3d(Z, Y, X):
    N = Z.shape[0]
    D = np.zeros((N,), dtype=np.float64)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            d = np.sqrt((Z[i] - Z[j])**2 + (Y[i] - Y[j])**2 + (X[i] - X[j])**2)
            D[j] += d
            D[i] += d
    return D

@jit(nopython=True)
def _distarray_nd(points):
    N = points.shape[0]
    ndim = points.shape[1]
    D = np.zeros((N,), dtype=np.float64)
    for i in range(0, N - 1):
        for j in range(i + 1, N):
            d2 = 0.0
            for k in range(ndim):
                d2 += (points[i, k] - points[j, k]) ** 2
            d = np.sqrt(d2)
            D[j] += d
            D[i] += d
    return D
