import numpy as np

def compute_edm(centers, output):
    """Compute euclidean distance map to nearest center.
    Works for 2D (Y, X) and 3D (Z, Y, X) output shapes.

    Args:
        centers: list of center coordinates, each with len matching output.ndim
        output: array to write into, shape (Y, X) or (Z, Y, X)
    """
    ndim = output.ndim
    centers = np.asarray(centers, output.dtype)
    # Build index grids: each has shape that broadcasts with (spatial..., num_centers)
    indices = []
    for dim in range(ndim):
        shape = [1] * (ndim + 1)  # +1 for centers axis
        shape[dim] = output.shape[dim]
        indices.append(np.arange(output.shape[dim], dtype=output.dtype).reshape(shape))
    # centers shape: (num_centers, ndim) -> broadcast dim at end
    squared_distances = sum(
        (idx - centers[:, dim]) ** 2
        for dim, idx in enumerate(indices)
    )
    min_squared_distances = np.min(squared_distances, axis=-1)
    np.sqrt(min_squared_distances, out=output)
