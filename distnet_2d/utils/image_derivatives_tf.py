import numpy as np
import tensorflow as tf
try:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
except:
    tfd = None


def der_2d(image, axis:int):
    """
        Compute the partial derivative (central difference approximation) of source in a particular dimension: d_f( x ) = ( f( x + 1 ) - f( x - 1 ) ) / 2.
        Output tensors has the same shape as the input: [B, Y, X, C].

        Args:
        image: Tensor with shape [B, Y, X, C].
        axis: axis to compute gradient on (1 = dy or 2 = dx)

        Returns:
        tensor dy or dx holding the vertical or horizontal partial derivative
        gradients (1-step finite difference).

        Raises:
        ValueError: If `image` is not a 4D tensor.
        """
    if isinstance(image, np.ndarray):
        assert image.ndim == 4, f'image_gradients expects a 4D tensor  [B, Y, X, C], not {image.shape}'
    else:
        tf.assert_equal(tf.rank(image), 4, message=f'image_gradients expects a 4D tensor  [B, Y, X, C], not {tf.shape(image)}.')
    assert axis in [1, 2], "axis must be in [1, 2]"
    if axis == 1:
        image = tf.pad(image, tf.constant([[0, 0], [1, 1,], [0, 0], [0, 0]]), mode="SYMMETRIC")
        return tf.math.divide(image[:, 2:, :, :] - image[:, :-2, :, :], tf.cast(2, image.dtype))
    else:
        image = tf.pad(image, tf.constant([[0, 0], [0, 0,], [1, 1], [0, 0]]), mode="SYMMETRIC")
        return tf.math.divide(image[:, :, 2:, :] - image[:, :, :-2, :], tf.cast(2, image.dtype))


def gradient_magnitude_2d(image=None, dy=None, dx=None, sqrt:bool=True):
    if image is None:
        assert dy is not None and dx is not None, "provide either image or partial derivatives"
        tf.assert_equal(tf.shape(dy), tf.shape(dx), message="partial derivatives must have same shape")
    else:
        dy = der_2d(image, 1)
        dx = der_2d(image, 2)

    grad = dx * dx + dy * dy
    if sqrt:
        grad = tf.math.sqrt(grad)
    return grad


def laplacian_2d(image=None, dy=None, dx=None):
    if image is None:
        assert dy is not None and dx is not None, "provide either image or partial derivatives"
        tf.assert_equal(tf.shape(dy), tf.shape(dx), message="partial derivatives must have same shape")
    else:
        dy = der_2d(image, 1)
        dx = der_2d(image, 2)

    ddy = der_2d(dy, 1)
    ddx = der_2d(dx, 2)
    return ddy + ddx


def smooth(image, rad:float=1.5):
    return convolve(image, kernel=make_gaussian_kernel_2d(rad))


def convolve(image, kernel, padding_mode:str='SYMMETRIC'):
    Y, X, _, _ = tf.unstack(tf.shape(kernel))
    assert Y % 2 == 1, "ker_size should be uneven in both spatial directions"
    assert X % 2 == 1, "ker_size should be uneven in both spatial directions"
    rY = (Y - 1) / 2
    rX = (Y - 1) / 2
    padded = tf.pad(image, [[0, 0], [rY, rY], [rX, rX], [0, 0]], padding_mode)
    return tf.nn.conv2d(padded, kernel, strides=1, padding='VALID')


def make_gaussian_kernel_2d(std, ker_size=None, n_chan:int=1):
    if ker_size is None:
        ker_size = max(3, (round(std)+1)*2+1)
    else:
        assert ker_size>=3, "ker_size should be >=3"
        assert ker_size%2==1, "ker_size should be uneven"
    extent = (ker_size-1)//2
    d = tfd.Normal(0, std)
    vals = d.prob(tf.range(start=-extent, limit=extent + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
    if n_chan>1:
        gauss_kernel = tf.tile(gauss_kernel, [1, 1, n_chan, 1])
    return gauss_kernel
