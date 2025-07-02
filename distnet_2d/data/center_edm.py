import numpy as np

def compute_edm(centers, output):
    Y, X = output.shape
    centers = np.asarray(centers, output.dtype)
    y_indices = np.arange(Y)[:, np.newaxis, np.newaxis]  # (Y, 1, 1)
    x_indices = np.arange(X)[np.newaxis, :, np.newaxis]  # (1, X, 1)
    dy = y_indices - centers[:, 0]  # (Y, 1, num_centers)
    dx = x_indices - centers[:, 1]  # (1, X, num_centers)
    squared_distances = dy**2 + dx**2  # (Y, X, num_centers)
    min_squared_distances = np.min(squared_distances, axis=2)  # (Y, X)
    np.sqrt(min_squared_distances, out = output)