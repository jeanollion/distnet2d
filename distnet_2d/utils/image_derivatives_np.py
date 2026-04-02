import numpy as np

def der(image, *axes:int):
    """
        Compute the partial derivative (central difference approximation) of source in a particular dimension: d_f( x ) = ( f( x + 1 ) - f( x - 1 ) ) / 2.
        Output tensor has the same shape as the input. Works for 2D and 3D images.

        Args:
        image: Tensor with shape [Y, X] or [Z, Y, X].
        axes: axes to compute gradient on

        Returns:
        tensor(s) holding the partial derivative gradients (1-step finite difference).
        If multiple axes, returns a list.
        """
    if len(axes) > 1:
        return [der(image, ax) for ax in axes]
    else:
        axis = axes[0]
    ndim = image.ndim
    assert 0 <= axis < ndim, f"axis {axis} out of range for {ndim}D image"
    pad_widths = [(0, 0)] * ndim
    pad_widths[axis] = (1, 1)
    image = np.pad(image, pad_widths, mode="edge")
    slc_hi = [slice(None)] * ndim
    slc_lo = [slice(None)] * ndim
    slc_hi[axis] = slice(2, None)
    slc_lo[axis] = slice(None, -2)
    return np.divide(image[tuple(slc_hi)] - image[tuple(slc_lo)], 2)


def der_2d(image, *axis:int):
    """Backward-compatible wrapper. See der()."""
    assert image.ndim == 2, f'image_gradients expects a 2D tensor [Y, X], not {image.shape}'
    return der(image, *axis)


def gradient_magnitude(image=None, derivatives=None, sqrt:bool=True):
    """Gradient magnitude for 2D or 3D images.

    Args:
        image: input image (2D or 3D). If None, derivatives must be provided.
        derivatives: list of partial derivatives [dy, dx] or [dz, dy, dx].
        sqrt: if True, return sqrt of sum of squares.
    """
    if image is None:
        assert derivatives is not None and len(derivatives) >= 2
    else:
        derivatives = der(image, *range(image.ndim))
    grad = sum(d * d for d in derivatives)
    if sqrt:
        grad = np.sqrt(grad)
    return grad


def gradient_magnitude_2d(image=None, dy=None, dx=None, sqrt:bool=True):
    """Backward-compatible wrapper."""
    derivatives = [dy, dx] if image is None else None
    return gradient_magnitude(image=image, derivatives=derivatives, sqrt=sqrt)


def laplacian(image=None, derivatives=None):
    """Laplacian for 2D or 3D images.

    Args:
        image: input image (2D or 3D). If None, derivatives must be provided.
        derivatives: list of partial derivatives.
    """
    if image is None:
        assert derivatives is not None and len(derivatives) >= 2
    else:
        derivatives = der(image, *range(image.ndim))
    return sum(der(d, axis) for axis, d in enumerate(derivatives))


def laplacian_2d(image=None, dy=None, dx=None):
    """Backward-compatible wrapper."""
    derivatives = [dy, dx] if image is None else None
    return laplacian(image=image, derivatives=derivatives)
