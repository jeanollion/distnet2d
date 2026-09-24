import tensorflow as tf
try:
    import tensorflow_probability as tfp
    tfd = tfp.distributions
except:
    tfd = None


def der(image, axis: int):
    """
        Compute partial derivative (central difference) along any axis. Works for any tensor rank.
    """
    ndim = len(image.shape)
    pad_widths = [[0, 0]] * ndim
    pad_widths[axis] = [1, 1]
    image = tf.pad(image, pad_widths, mode="SYMMETRIC")
    slc_hi = [slice(None)] * ndim
    slc_lo = [slice(None)] * ndim
    slc_hi[axis] = slice(2, None)
    slc_lo[axis] = slice(None, -2)
    return tf.math.divide(image[tuple(slc_hi)] - image[tuple(slc_lo)], tf.cast(2, image.dtype))


def laplacian(image=None, derivatives=None):
    """Compute Laplacian for nD tensors. Supports 4D (B,Y,X,C) and 5D (B,Z,Y,X,C).
    derivatives: list of first-order spatial derivatives [dz, dy, dx] or [dy, dx]."""
    if image is not None:
        ndim = len(image.shape)
        if ndim == 5:
            derivatives = [der(image, 1), der(image, 2), der(image, 3)]
        else:
            derivatives = [der(image, 1), der(image, 2)]
    assert derivatives is not None
    if len(derivatives) == 3:
        return der(derivatives[0], 1) + der(derivatives[1], 2) + der(derivatives[2], 3)
    else:
        return der(derivatives[0], 1) + der(derivatives[1], 2)


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
