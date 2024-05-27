import numpy as np

def der_2d(image, *axis:int):
    """
        Compute the partial derivative (central difference approximation) of source in a particular dimension: d_f( x ) = ( f( x + 1 ) - f( x - 1 ) ) / 2.
        Output tensors has the same shape as the input: [Y, X].

        Args:
        image: Tensor with shape [Y, X].
        axis: axis to compute gradient on (0 = dy or 1 = dx)

        Returns:
        tensor dy or dx holding the vertical or horizontal partial derivative
        gradients (1-step finite difference).

        Raises:
        ValueError: If `image` is not a 2D tensor.
        """
    if len(axis) > 1:
        return [der_2d(image, ax) for ax in axis]
    else:
        axis = axis[0]
    assert image.ndim == 2, f'image_gradients expects a 2D tensor  [Y, X], not {image.shape}'
    assert axis in [0, 1], "axis must be in [0, 1]"
    if axis == 0:
        image = np.pad(image, ((1, 1), (0, 0)), mode="edge")
        return np.divide(image[2:] - image[:-2], 2)
    else:
        image = np.pad(image, ((0, 0), (1, 1)), mode="edge")
        return np.divide(image[:, 2:] - image[:, :-2], 2)


def gradient_magnitude_2d(image=None, dy=None, dx=None, sqrt:bool=True):
    if image is None:
        assert dy is not None and dx is not None, "provide either image or partial derivatives"
        assert dy.shape == dx.shape, "partial derivatives must have same shape"
    else:
        dy, dx = der_2d(image, 0, 1)
    grad = dx * dx + dy * dy
    if sqrt:
        grad = np.sqrt(grad)
    return grad


def laplacian_2d(image=None, dy=None, dx=None):
    if image is None:
        assert dy is not None and dx is not None, "provide either image or partial derivatives"
        assert dy.shape == dx.shape, "partial derivatives must have same shape"
    else:
        dy, dx = der_2d(image, 0, 1)
    ddy = der_2d(dy, 0)
    ddx = der_2d(dx, 1)
    return ddy + ddx
