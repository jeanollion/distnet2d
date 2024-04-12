import tensorflow as tf
import numpy as np
from .helpers import ensure_multiplicity

# assumes iterator in return_central_only= True (thus framewindow = 1 and next = true)
def get_metrics_fun(spatial_dims, center_scale:float, max_objects_number:int=0, reduce:bool=True):
    spatial_dims = ensure_multiplicity(2, spatial_dims)
    scale = tf.cast(center_scale, tf.float32)
    coord_distance_fun = _coord_distance_fun()
    spa_wmean_fun = _get_spatial_wmean_by_object_fun(*spatial_dims)
    mean_fun = _get_mean_by_object_fun()
    mean_fun_lm = _get_mean_by_object_fun(nan=-1)
    def fun(args):
        edm, gdcm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
        labels = tf.transpose(labels, perm=[2, 0, 1]) # T, Y, X
        gdcm = tf.transpose(gdcm, perm=[2, 0, 1])
        edm = tf.transpose(edm, perm=[2, 0, 1])
        true_edm = tf.transpose(true_edm, perm=[2, 0, 1])
        motion_shape = tf.shape(dY)
        lm = tf.reshape(lm, shape=tf.concat([motion_shape[:2], motion_shape[-1:], [3] ], 0))
        lm = tf.transpose(lm, perm=[2, 0, 1, 3]) # T, Y, X, 3
        true_lm = tf.transpose(true_lm, perm=[2, 0, 1])
        ids, sizes, N = _get_label_size(labels, max_objects_number) # (T, N), (T, N)
        true_center_ob = true_center_ob[:, :N]

        # EDM : foreground/background IoU + contour IoU
        zero = tf.cast(0, edm.dtype)
        one = tf.cast(1, edm.dtype)
        pred_foreground = tf.math.greater(edm, zero)
        true_foreground = tf.math.greater(labels, 0)
        edm_IoU = _IoU(true_foreground, pred_foreground, tolerance=True)

        pred_contours = tf.math.logical_and( tf.math.greater(edm, zero), tf.math.less_equal(edm, tf.cast(1.5, edm.dtype)) )
        true_contours = tf.math.logical_and( tf.math.greater(true_edm, zero), tf.math.less_equal(true_edm, one) )
        contour_IoU = _IoU(true_contours, pred_contours, tolerance=True)
        edm_IoU = 0.5 * (edm_IoU + contour_IoU)

        #mask = tf.where(labels > 0, one, zero)
        #edm_L2 = tf.math.divide_no_nan(tf.math.reduce_sum(mask * (edm - true_edm) ** 2), tf.cast(tf.reduce_sum(sizes), edm.dtype))

        # CENTER  compute center coordinates per objects: spatial mean of predicted gaussian function of GDCM
        center_values = tf.math.exp(-tf.math.square(tf.math.divide(gdcm, scale)))
        center_coord = _objectwise_compute(center_values, [0], spa_wmean_fun, labels, ids, sizes) # (N, 2)
        #print(f"center: {tf.concat([true_center_ob[0], center_coord[0]], -1).numpy()}")
        center_l2 = coord_distance_fun(true_center_ob, center_coord)
        center_l2 = tf.cond(tf.math.is_nan(center_l2), lambda: tf.cast(0, center_l2.dtype), lambda: center_l2)

        # motion: l2 of pred vs true center coordinates
        dYX = tf.stack([dY, dX], -1) # Y, X, T, 2
        dYX = tf.transpose(dYX, perm=[2, 0, 1, 3]) # T, Y, X, 2
        #print(f"dXY shape: {dYX.shape} dY: {dY.shape}")
        dm = _objectwise_compute(dYX, [0, 1], mean_fun, labels, ids, sizes, label_channels=[0, 0])

        true_dYX = tf.stack([true_dY, true_dX], -1)  # Y, X, T, 2
        true_dYX = tf.transpose(true_dYX, perm=[2, 0, 1, 3])  # T, Y, X, 2
        true_dm = _objectwise_compute(true_dYX, [0, 1], mean_fun, labels, ids, sizes, label_channels=[0, 0])
        #print(f"dM: {tf.concat([true_dm[0], dm[0]], -1).numpy()}")
        #print(f"NEXT: dM: {tf.concat([true_dm[1], dm[1]], -1).numpy()}")
        dm_l2 = coord_distance_fun(true_dm, dm)
        dm_l2 = tf.cond(tf.math.is_nan(dm_l2), lambda: tf.cast(0, dm_l2.dtype), lambda: dm_l2)

        # Link Multiplicity
        true_lm = tf.cast(_objectwise_compute(true_lm[..., tf.newaxis], [0, 1], mean_fun_lm, labels, ids, sizes, label_channels=[0, 0]), tf.int32)[...,0]-1
        lm = _objectwise_compute(lm, [0, 1], mean_fun_lm, labels, ids, sizes, label_channels=[0, 0])
        lm = tf.math.argmax(lm, axis=-1, output_type=tf.int32)
        #print(f"lm: {tf.stack([true_lm[0], lm[0]], -1).numpy()}")
        #print(f"NEXT lm: {tf.stack([true_lm[1], lm[1]], -1).numpy()}")
        errors = tf.math.not_equal(lm, true_lm)
        lm_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
        return edm_IoU, -center_l2, -dm_l2, -lm_errors

    def metrics_fun(edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
        metrics = tf.map_fn(fun, (edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32) )
        if reduce:
            metrics = [tf.reduce_mean(m) for m in metrics]
        return metrics
    return metrics_fun


def _get_label_size(labels, max_objects_number:int=0): # C, Y, X
    N = max_objects_number if max_objects_number>0 else tf.math.reduce_max(labels)
    def treat_image(im):
        ids, _, counts = tf.unique_with_counts(im)
        if tf.math.equal(tf.shape(ids)[0], 1) and tf.math.equal(ids[0], 0): # null case: only zeros
            ids = tf.zeros(shape = (N,), dtype=tf.int32)
            counts = tf.zeros(shape = (N,), dtype=tf.int32)
        else:
            non_null = tf.math.not_equal(ids, 0)
            ids = tf.boolean_mask(ids, non_null)
            counts = tf.boolean_mask(counts, non_null)
            indices = ids - 1
            indices = indices[...,tf.newaxis]
            ids = tf.scatter_nd(indices, ids, shape = (N,))
            counts = tf.scatter_nd(indices, counts, shape = (N,))
        return ids, counts
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1]) # (C, Y*X)
    ids, size = tf.map_fn(treat_image, labels, fn_output_signature=(tf.int32, tf.int32))
    return ids, size, N

def _get_spatial_wmean_by_object_fun(Y, X):
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

def _get_mean_by_object_fun(nan=float('NaN')):
    nan = tf.cast(nan, tf.float32)
    def fun(data, mask, size): # (Y, X, C)
        mask = tf.expand_dims(mask, -1)
        nan_tensor = tf.repeat(nan, repeats=tf.shape(data)[-1])
        non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
        return tf.cond(tf.math.equal(size, 0), lambda:nan_tensor, non_null)
    return fun

def _objectwise_compute(data, channels, fun, labels, ids, sizes, label_channels=None): # [(tensor, range, fun)], (T, Y, X), (T, N), (T, N)
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
        mask = tf.cond(tf.math.equal(id, 0), lambda:0., lambda:tf.cast(tf.math.equal(labels, id), tf.float32))
        return fun(data, mask, size)
    return tf.map_fn(treat_ob, (ids, sizes), fn_output_signature=data.dtype)

def _coord_distance_fun():
    def loss(true, pred): # (C, N, 2)
        no_na_mask = tf.stop_gradient(tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,:1]), tf.math.is_nan(pred[...,:1]))), tf.float32)) # non-empty objects # (C, N, 1)
        n_obj = tf.reduce_sum(no_na_mask[...,0], axis=-1, keepdims=False) # (C)
        true = tf.math.multiply_no_nan(true, no_na_mask)  # (C, N, 2)
        pred = tf.math.multiply_no_nan(pred, no_na_mask)  # (C, N, 2)
        d = tf.math.square(true - pred)  # (C, N, 2) # TODO for a loss function use Huber loss
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, N)
        #print(f"n_obj: {n_obj} \ndistances: \n{d} \nsize: \n{size}")
        #d = tf.math.divide_no_nan(d, size)
        #return tf.math.reduce_sum(d, keepdims=False)
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C)
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

def _IoU(true, pred, tolerance:bool=False):
    true_inter = _dilate_mask(true) if tolerance else true
    intersection = tf.math.count_nonzero(tf.math.logical_and(true_inter, pred))
    union = tf.math.count_nonzero(tf.math.logical_or(true, pred))
    return tf.math.divide_no_nan(tf.cast(intersection, tf.float32), tf.cast(union, tf.float32))

def _dilate_mask(maskYX):
    maskYX = tf.cast(maskYX, tf.float16)
    mean_kernel = [[1./9, 1./9, 1./9], [1./9, 1./9, 1./9], [1./9, 1./9, 1./9]]
    conv = _convolve(maskYX, tf.cast(mean_kernel, tf.float16))
    return tf.math.greater(conv, tf.cast(0., tf.float16))

def _contour_IoU_fun(pred_contour, mask, size):
    true_contours = _compute_contours(mask)
    return _IoU(true_contours, pred_contour)

def _compute_contours(maskYX):
    kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    conv = _convolve(maskYX, kernel)
    return tf.math.greater(conv, 0) # detect at least one zero in the neighborhood

def _convolve(imageYX, kernel):
    padded = tf.pad(imageYX, [[0, 0], [1, 1], [1, 1]], 'SYMMETRIC')
    input = padded[..., tf.newaxis]
    conv = tf.nn.conv2d(input, kernel[:, :, tf.newaxis, tf.newaxis], strides=1, padding='VALID')
    return conv[..., 0]