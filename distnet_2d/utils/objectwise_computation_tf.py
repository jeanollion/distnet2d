import tensorflow as tf
import numpy as np


def get_label_size(labels, max_objects_number:int=0): # C, Y, X
    N = max_objects_number if max_objects_number>0 else tf.math.maximum(tf.cast(1, labels.dtype), tf.math.reduce_max(labels))

    def treat_image(im):
        def non_null(ids, counts):
            non_null_ids = tf.math.not_equal(ids, 0)
            ids = tf.boolean_mask(ids, non_null_ids)
            counts = tf.boolean_mask(counts, non_null_ids)
            indices = ids - 1
            indices = indices[..., tf.newaxis]
            ids = tf.scatter_nd(indices, ids, shape=(N,))
            counts = tf.scatter_nd(indices, counts, shape=(N,))
            return ids, counts

        def null():
            return tf.zeros(shape=(N,), dtype=tf.int32), tf.zeros(shape=(N,), dtype=tf.int32)

        ids, _, counts = tf.unique_with_counts(im)
        return tf.cond(tf.math.logical_and(tf.math.equal(tf.shape(ids)[0], 1), tf.math.equal(ids[0], 0)), null, lambda: non_null(ids, counts))

    labels = tf.reshape(labels, [tf.shape(labels)[0], -1]) # (C, Y*X)
    ids, size = tf.map_fn(treat_image, labels, fn_output_signature=(tf.int32, tf.int32))
    return ids, size, N


def get_spatial_wmean_by_object_fun(Y, X):
    Y, X = tf.meshgrid(tf.range(Y, dtype=tf.float32), tf.range(X, dtype=tf.float32), indexing='ij')
    nan = tf.cast(float('NaN'), tf.float32)

    def apply(data, mask, size):
        def non_null():
            data_masked = tf.math.multiply_no_nan(data, mask)
            wsum_y = tf.reduce_sum(data_masked * Y, keepdims=False)
            wsum_x = tf.reduce_sum(data_masked * X, keepdims=False)
            wsum = tf.stack([wsum_y, wsum_x]) # (2)
            sum = tf.reduce_sum(data_masked, keepdims = False)
            #sum = tf.stop_gradient(sum)
            return tf.math.divide(wsum, sum) # when no values should return nan # (2)
        return tf.cond(tf.math.equal(size, 0), lambda:tf.stack([nan, nan]), non_null)
    return apply


def get_soft_argmax_2d_by_object_fun(Y, X, beta=1e2):
    Y, X = tf.meshgrid(tf.range(Y, dtype=tf.float32), tf.range(X, dtype=tf.float32), indexing='ij')
    nan = tf.cast(float('NaN'), tf.float32)

    def sam(data, mask, size): # (Y, X)
        def non_null():
            shape = tf.shape(data)
            data_masked = tf.math.multiply_no_nan(data, mask)
            data_masked = tf.reshape(data_masked, (-1,)) # (X * Y,)
            data_masked = tf.nn.softmax(data_masked * beta, axis=0)
            data_masked = tf.reshape(data_masked, shape) # (X, Y)
            argmax_y = tf.reduce_sum(data_masked * Y, axis=[0, 1], keepdims=False) # (1,)
            argmax_x = tf.reduce_sum(data_masked * X, axis=[0, 1], keepdims=False) # (1,)
            argmax = tf.stack([argmax_y, argmax_x], -1) # (2,)
            sum = tf.reduce_sum(data_masked, keepdims=False)
            # sum = tf.stop_gradient(sum)
            return tf.math.divide(argmax, sum)  # when no values should return nan # (2)
        return tf.cond(tf.math.equal(size, 0), lambda:tf.stack([nan, nan]), non_null) # when no values should return nan
    return sam


def get_argmax_2d_by_object_fun(nan=float('NaN')):
    nan = tf.cast(nan, tf.float32)
    def fun(data, mask, size): # (Y, X)
        def non_null():
            shape = tf.shape(data)
            data_masked = tf.math.multiply_no_nan(data, mask)
            data_masked = tf.reshape(data_masked, (-1,)) # (X * Y,)
            idx_max = tf.math.argmax(data_masked, axis=0, output_type=tf.dtypes.int32)
            return tf.cast(tf.unravel_index(idx_max, dims=shape), tf.float32)
        return tf.cond(tf.math.equal(size, 0), lambda:tf.stack([nan, nan]), non_null) # when no values should return nan
    return fun


def get_mean_by_object_fun(nan=float('NaN'), channel_axis:bool=True):
    nan = tf.cast(nan, tf.float32)
    if channel_axis:
        def fun(data, mask, size): # (Y, X, C), (Y, X), (1,) -> (C,)
            mask = tf.expand_dims(mask, -1)
            null = lambda: tf.repeat(nan, repeats=tf.shape(data)[-1])
            non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun
    else:
        def fun(data, mask, size): # (Y, X), (Y, X), (1,) -> (1,)
            null = lambda: nan
            non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun


def get_max_by_object_fun(nan=float('NaN'), channel_axis:bool=True):
    nan = tf.cast(nan, tf.float32)
    if channel_axis:
        def fun(data, mask, size):  # (Y, X, C), (Y, X), (1,) -> (1,)
            mask = tf.expand_dims(mask, -1)
            null = lambda: tf.repeat(nan, repeats=tf.shape(data)[-1])
            non_null = lambda: tf.reduce_max(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False)
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun
    else:
        def fun(data, mask, size):  # (Y, X), (Y, X), (1,)
            null = lambda: nan
            non_null = lambda: tf.reduce_max(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False)
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun


def objectwise_compute(data, fun, labels, ids, sizes): # tensor (Y, X, ...) , fun, (Y, X), (N,), (N,) -> (N, ...)

    def non_null():
        ta = tf.TensorArray(dtype=data.dtype, size=tf.shape(ids)[0])
        for i in tf.range(tf.shape(ids)[0]):
            mask = tf.cast(tf.math.equal(labels, ids[i]), tf.float32)
            ta = ta.write(i, fun(data, mask, sizes[i]))
        return ta.stack()

    def null():
        return fun(data, tf.zeros_like(labels, dtype=tf.float32), 0)
    return tf.cond(tf.math.equal(tf.size(ids), 0), null, non_null)


def objectwise_compute_channel(data, fun, labels, ids, sizes): # tensor (C, Y, X, ...) , fun, (Y, X), (N), ( N) -> (C, N, ...)

    def non_null():
        n_chan = tf.shape(data)[0]
        n_obj = tf.shape(ids)[0]
        ta = tf.TensorArray(dtype=data.dtype, size=n_obj * n_chan)
        for i in tf.range(n_obj):
            mask = tf.cast(tf.math.equal(labels, ids[i]), tf.float32)
            for j in tf.range(n_chan):
                ta = ta.write(j * n_obj + i, fun(data[j], mask, sizes[i]))
        tensor = ta.stack()
        return tf.reshape(tensor, shape=tf.concat([[n_chan, n_obj], tf.shape(tensor)[1:]], 0))

    def null():
        n_chan = tf.shape(data)[0]
        ta = tf.TensorArray(dtype=data.dtype, size=n_chan)
        mask = tf.zeros_like(labels, dtype=tf.float32)
        for j in tf.range(n_chan):
            ta = ta.write(j, fun(data[j], mask, 0))
        tensor = ta.stack()
        return tf.reshape(tensor, shape=tf.concat([[n_chan, 1], tf.shape(tensor)[1:]], 0))
    return tf.cond(tf.math.equal(tf.size(ids), 0), null, non_null)


def coord_distance_fun(max:bool=True, sqrt:bool=False, pop_fraction=0.25):
    def loss(true, pred): # (C, N, 2)
        no_na_mask = tf.stop_gradient(tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,:1]), tf.math.is_nan(pred[...,:1]))), tf.float32)) # non-empty objects # (C, N, 1)
        n_obj = tf.reduce_sum(no_na_mask[...,0], axis=-1, keepdims=False) # (C, )
        true = tf.math.multiply_no_nan(true, no_na_mask)  # (C, N, 2)
        pred = tf.math.multiply_no_nan(pred, no_na_mask)  # (C, N, 2)
        d = tf.math.square(true - pred)  # (C, N, 2) # to use this function as a loss function use Huber loss instead of L2
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, N)
        if sqrt:
            d = tf.math.sqrt(d)
        if max:
            if pop_fraction <= 0:
                return tf.math.reduce_max(d)
            else:
                d = tf.map_fn(lambda args: reduce_pop_size(tensor=args[0], N=args[1], pop_fraction=pop_fraction), (d, n_obj), fn_output_signature=tf.float32)
                return tf.math.reduce_max(d)  # max over channel
        else: # mean
            d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, )
            d = tf.math.divide_no_nan(d, n_obj) # mean over objects
            return tf.math.reduce_mean(d) # mean over channel
    return loss

# for small population: returns max, for large population returns mean of top_k
def reduce_pop_size(tensor, N, pop_fraction:float=0.25):
    N = tf.cast(N, tf.int32)
    pop_fraction = tf.cast(pop_fraction, tf.float32)
    pop_size_limit = tf.cast(tf.math.ceil(2 / pop_fraction), tf.int32)

    small_pop = lambda: tf.reduce_max(tensor)

    def large_pop():
        top_k, _ = tf.math.top_k(tensor, k=tf.cast(tf.cast(N, tf.float32) * pop_fraction, tf.int32))
        return tf.reduce_mean(top_k)

    return tf.cond(tf.math.greater_equal(N, pop_size_limit), large_pop, small_pop)

def _get_spatial_kernels(Y, X, two_channel_axes=True, batch_axis=True):
    Y, X = tf.meshgrid(tf.range(Y, dtype = tf.float32), tf.range(X, dtype = np.float32), indexing = 'ij')
    if batch_axis:
        Y, X = Y[tf.newaxis], X[tf.newaxis]
    if two_channel_axes:
        return Y[..., tf.newaxis, tf.newaxis], X[..., tf.newaxis, tf.newaxis]
    else:
        return Y[..., tf.newaxis], X[..., tf.newaxis]


def _generate_kernel(sizeY, sizeX, C=1, O=0):
    kernel = np.meshgrid(np.arange(sizeY, dtype = np.float32), np.arange(sizeX, dtype = np.float32), indexing = 'ij') #(Y,X), (Y,X)
    kernel = np.stack(kernel, axis=-1) # (Y, X, 2)
    kernel = np.expand_dims(kernel, -2) # (Y, X, 1, 2)
    if C is not None and C>1:
        kernel = np.tile(kernel,[1,1,C,1]) # (Y, X, C, 2)
    if O is not None and O>0: # add object dimension
        kernel = np.expand_dims(kernel, -2) # (Y, X, nC, 1, 2)
        if O>1:
            rep = [1]*np.rank(kernel)
            rep[-2] = O
            kernel = np.tile(kernel,rep) # (Y, X, C, O, 2)
    return kernel


def IoU(true_foreground, pred_foreground, tolerance_radius:float=0):
    true_inter = _dilate_mask(true_foreground, radius=tolerance_radius, symmetric_padding=True) if tolerance_radius>=1 else true_foreground
    intersection = tf.math.count_nonzero(tf.math.logical_and(true_inter, pred_foreground), keepdims=False)
    union = tf.math.count_nonzero(tf.math.logical_or(true_foreground, pred_foreground), keepdims=False)
    return tf.cond(tf.math.equal(union, tf.cast(0, union.dtype)), lambda: tf.cast(1., tf.float32), lambda: tf.math.divide(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32)))  # if union is null -> metric is 1


def FP(true_foreground, pred_foreground, rate:bool = False, tolerance_radius:float=0):
    true_background = tf.math.logical_not(true_foreground)
    false_positives = tf.logical_and(pred_foreground, true_background)
    false_positives = _erode_mask(false_positives, radius=tolerance_radius, symmetric_padding=False) if tolerance_radius>=1 else false_positives
    fp = tf.math.count_nonzero(false_positives, keepdims=False)
    if rate: # FPR
        tn = tf.math.count_nonzero(true_background, keepdims=False) # for FRP
        return tf.math.divide_no_nan(tf.cast(fp, tf.float32), tf.cast(tn, tf.float32))
    else: # FPD
        #return tf.cast(fp, tf.float32)
        npix = tf.reduce_prod(tf.shape(true_background)) # for FPD
        return tf.math.divide(tf.cast(fp, tf.float32), tf.cast(npix, tf.float32))


def _dilate_mask(maskBYX, radius:float=1.5, tolerance:float=0.25, symmetric_padding:bool=True):
    assert 0<=tolerance<0.5
    maskBYX = tf.cast(maskBYX, tf.int32)
    ker, rad = circular_kernel(radius)
    thld = tf.math.floor(tf.cast(tf.math.reduce_sum(ker), tf.float32) * tf.cast(tolerance, tf.float32))
    conv = _convolve(maskBYX, ker, rad, symmetric_padding=symmetric_padding)
    return tf.math.greater(conv, tf.cast(thld, tf.int32))


def _erode_mask(maskBYX, radius:float=1.5, tolerance:float=0.25, symmetric_padding:bool=False):
    assert 0 <= tolerance < 0.5
    maskBYX = tf.cast(maskBYX, tf.int32)
    ker, rad = circular_kernel(radius)
    thld = tf.math.ceil(tf.cast(tf.math.reduce_sum(ker), tf.float32) * tf.cast(1 - tolerance, tf.float32))
    conv = _convolve(maskBYX, ker, rad, symmetric_padding=symmetric_padding)
    return tf.math.greater_equal(conv, tf.cast(thld, tf.int32))



def circular_kernel(radius: float) :
    """
    Create a circular 2D kernel of ones with a given float radius.
    Args:
        radius: The radius of the circle (float).
    Returns:
        A 2D TensorFlow tensor representing the circular kernel (dtype: tf.int32).
    """
    radius_int = tf.cast(radius, tf.int32)
    diameter = tf.cast(2 * radius_int + 1, tf.int32)
    center = diameter // 2

    # Create a grid of coordinates
    y = tf.range(-center, diameter - center, dtype=tf.float32)
    x = tf.range(-center, diameter - center, dtype=tf.float32)
    y_grid, x_grid = tf.meshgrid(y, x, indexing='ij')

    # Compute the distance from the center
    distance = tf.math.sqrt(x_grid**2 + y_grid**2)

    # Create the circular kernel
    kernel = tf.zeros((diameter, diameter), dtype=tf.int32)
    kernel = tf.where(distance <= radius, tf.ones_like(kernel, dtype=tf.int32), kernel)

    return kernel, radius_int


def _contour_IoU_fun(pred_contour, mask, size):
    true_contours = _compute_contours(mask)
    return IoU(true_contours, pred_contour)


def _compute_contours(maskBYX):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    conv = _convolve(maskBYX, kernel, 1, symmetric_padding=True)
    return tf.math.greater(conv, tf.cast(0, conv.dtype)) # detect at least one zero in the neighborhood


def _convolve(imageBYX, kernel, radius, symmetric_padding:bool):
    if symmetric_padding:
        imageBYX = tf.pad(imageBYX, [[0, 0], [radius, radius], [radius, radius]], 'SYMMETRIC')
    imageBYX = imageBYX[..., tf.newaxis]
    kernel = tf.cast(kernel, imageBYX.dtype)
    conv = tf.nn.conv2d(imageBYX, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='VALID' if symmetric_padding else "SAME")
    return conv[..., 0]