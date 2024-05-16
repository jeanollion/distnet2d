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
        def fun(data, mask, size): # (Y, X, C), (Y, X), (1,)
            mask = tf.expand_dims(mask, -1)
            null = lambda: tf.repeat(nan, repeats=tf.shape(data)[-1])
            non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun
    else:
        def fun(data, mask, size): # (Y, X), (Y, X), (1,)
            null = lambda: nan
            non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
            return tf.cond(tf.math.equal(size, 0), null, non_null)
        return fun


def get_max_by_object_fun(nan=float('NaN'), channel_axis:bool=True):
    nan = tf.cast(nan, tf.float32)
    if channel_axis:
        def fun(data, mask, size):  # (Y, X, C), (Y, X), (1,)
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


def objectwise_compute(data, channels, fun, labels, ids, sizes, label_channels=None): # [(tensor, range, fun)], (T, Y, X (,C) ), (T, N), (T, N)
    if label_channels is None:
        label_channels = channels
    else:
        assert len(label_channels) == len(channels), "label_channels and channels must have same length"

    def treat_im(args):
        dc, lc = args
        return _objectwise_compute_channel(data[dc], fun, labels[lc], ids[lc], sizes[lc])
    return tf.map_fn(treat_im, (tf.convert_to_tensor(channels), tf.convert_to_tensor(label_channels)), fn_output_signature=data.dtype)


def _objectwise_compute_channel(data, fun, labels, ids, sizes): # tensor, fun, (Y, X), (N), ( N)
    def treat_ob(args):
        id, size = args
        mask = tf.cond(tf.math.equal(id, 0), lambda:tf.zeros_like(labels, dtype=tf.float32), lambda:tf.cast(tf.math.equal(labels, id), tf.float32))
        return fun(data, mask, size)
    return tf.map_fn(treat_ob, (ids, sizes), fn_output_signature=data.dtype)


def coord_distance_fun(max:bool=True, sqrt:bool=False):
    def loss(true, pred): # (C, N, 2)
        no_na_mask = tf.stop_gradient(tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,:1]), tf.math.is_nan(pred[...,:1]))), tf.float32)) # non-empty objects # (C, N, 1)
        n_obj = tf.reduce_sum(no_na_mask[...,0], axis=-1, keepdims=False) # (C)
        true = tf.math.multiply_no_nan(true, no_na_mask)  # (C, N, 2)
        pred = tf.math.multiply_no_nan(pred, no_na_mask)  # (C, N, 2)
        d = tf.math.square(true - pred)  # (C, N, 2) # TODO for a loss function use Huber loss
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, N)
        if sqrt:
            d = tf.math.sqrt(d)
        #print(f"n_obj: {n_obj} \ndistances: \n{d} \nsize: \n{size}")
        #d = tf.math.divide_no_nan(d, size)
        #return tf.math.reduce_sum(d, keepdims=False)
        if max:
            return tf.math.reduce_max(d)
        else:
            d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, )
            d = tf.math.divide_no_nan(d, n_obj) # mean over objects
            return tf.math.reduce_mean(d) # mean over channel
    return loss


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


def IoU(true, pred, tolerance:bool=False):
    true_inter = _dilate_mask(true) if tolerance else true
    intersection = tf.math.count_nonzero(tf.math.logical_and(true_inter, pred), keepdims=False)
    union = tf.math.count_nonzero(tf.math.logical_or(true, pred), keepdims=False)
    return tf.cond(tf.math.equal(union, tf.cast(0, union.dtype)), lambda: tf.cast(1., tf.float32), lambda: tf.math.divide(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32)))


def _dilate_mask(maskYX):
    maskYX = tf.cast(maskYX, tf.float16)
    mean_kernel = [[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]]
    conv = _convolve(maskYX, tf.cast(mean_kernel, tf.float16))
    return tf.math.greater(conv, tf.cast(0., tf.float16))


def _contour_IoU_fun(pred_contour, mask, size):
    true_contours = _compute_contours(mask)
    return IoU(true_contours, pred_contour)


def _compute_contours(maskYX):
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    conv = _convolve(maskYX, kernel)
    return tf.math.greater(conv, 0) # detect at least one zero in the neighborhood


def _convolve(imageYX, kernel):
    padded = tf.pad(imageYX, [[0, 0], [1, 1], [1, 1]], 'SYMMETRIC')
    input = padded[..., tf.newaxis]
    conv = tf.nn.conv2d(input, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='VALID')
    return conv[..., 0]