import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError

def get_soft_argmax_2d_fun(beta=1e2, two_channel_axes=True):
    @tf.function
    def sam(x, x_flat=None, label_rank=None):
        shape = tf.shape(x)
        Y, X = _get_spatial_kernels(shape[1], shape[2], two_channel_axes)
        count = tf.expand_dims(tf.math.count_nonzero(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, T, C, 1)
        x = tf.reshape(x, tf.concat([shape[:1], [-1], shape[-1:]], 0))
        x = tf.nn.softmax(x * beta, axis = 1)
        x = tf.reshape(x, shape)
        x = label_rank * x
        argmax_y = tf.reduce_sum(x*Y, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        argmax_x = tf.reduce_sum(x*X, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        argmax = tf.stack([argmax_y, argmax_x], -1) # (B, 1, 1, T, C, 2)
        return tf.where(count==0, tf.constant(float('NaN')), argmax) # when no values should return nan
    return sam

def get_edm_max_2d_fun(tolerance = 0.9, two_channel_axes=True):
    tolerance = tf.cast(tolerance, tf.float32)
    zero = tf.cast(0, tf.float32)
    wm = get_weighted_mean_2d_fun(two_channel_axes)
    @tf.function
    def sam(x, x_flat=None, label_rank=None):
        edm_max = tf.math.reduce_max(x, axis=[1, 2], keepdims = True) * tolerance
        mask = tf.cast(tf.greater_equal(x, edm_max), tf.float32)
        return wm(x * mask)
    return sam

def get_weighted_mean_2d_fun(two_channel_axes=True):
    @tf.function
    def wm(x, x_flat=None, label_rank=None):
        shape = tf.shape(x)
        Y, X = _get_spatial_kernels(shape[1], shape[2], two_channel_axes)
        wsum_y = tf.reduce_sum(x * Y, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        wsum_x = tf.reduce_sum(x * X, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        wsum = tf.stack([wsum_y, wsum_x], -1) # (B, 1, 1, T, C, 2)
        sum = tf.expand_dims(tf.reduce_sum(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, T, C, 1)
        return tf.math.divide(wsum, sum) # when no values should return nan # (B, 1, 1, T, C, 2)
    return wm

def get_skeleton_center_fun(w = 0.1):
    wm = get_weighted_mean_2d_fun(True)
    @tf.function
    def sk(edm_ob, edm, label_rank): # (B, Y, X, T, N), (B, Y, X, T), (B, Y, X, T, N)
        shape = tf.shape(edm)
        Y, X = _get_spatial_kernels(shape[1], shape[2], True)
        center = wm(edm_ob) # B, 1, 1, T, N, 2
        mp = tf.nn.max_pool(edm, ksize=3, strides=1, padding="SAME")
        lm = tf.cast(tf.math.equal(mp, edm), tf.float32) # TODO: also limit to edm > 1 ?
        lm = label_rank * tf.expand_dims(lm, -1) # B, Y, X, T, N
        lm_coords = tf.stack([lm * Y, lm * X], -1) # B, Y, X, T, N, 2
        lm_dist = tf.math.reduce_sum(tf.math.square(lm_coords - center), [-1], keepdims = False) # B, Y, X, T, N
        lm_dist_inv = tf.math.divide_no_nan(1., lm_dist + w)
        lm_dist_inv = lm * lm_dist_inv # mask other than lm
        return wm(lm_dist_inv)
    return sk

def get_gaussian_spread_fun(sigma, Y, X, objectwise=True):
    kernel = _generate_kernel(Y, X, O=1 if objectwise else 0) # (Y, X, 1, 2) or # (Y, X, 1, 1, 2)
    kernel = tf.constant(np.expand_dims(kernel, 0), dtype=tf.float32) # (1, Y, X, 1, 2) or  (1, Y, X, 1, 1, 2)
    mul = -1./(sigma * sigma)
    zero = tf.constant(0, dtype=tf.float32)

    @tf.function
    def gs(x): # (B, 1, 1, C, 2) or (B, 1, 1, C, N, 2)
        d = tf.math.reduce_sum(tf.math.square(x - kernel), axis=-1, keepdims=False) #(B, Y, X, C) or (B, Y, X, C, N)
        exp = tf.math.exp(d * mul)
        exp = tf.where(tf.math.is_nan(exp), zero, exp)
        if objectwise:
            exp = tf.reduce_sum(exp, -1) #(B, Y, X, C)
        return exp
    return gs

def get_euclidean_distance_loss(image_shape, objectwise=True):
    sum_axis = [-1, -2] if objectwise else [-1]
    im_scale = tf.cast(1./(image_shape[0]*image_shape[1]), tf.float32)
    def loss(true, pred, object_size=None): # (B, 1, 1, C, N, 2) or (B, 1, 1, C, 2), and (B, 1, 1, C, N)
        no_na_mask = tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,:1]), tf.math.is_nan(pred[...,:1]))), tf.float32) # non-empty objects
        true = tf.math.multiply_no_nan(true, no_na_mask)
        pred = tf.math.multiply_no_nan(pred, no_na_mask)
        d = tf.math.reduce_sum(tf.math.square(true - pred), axis=sum_axis, keepdims=False) #(B, 1, 1, C)
        if objectwise: # normalize by object size / image size
            n_obj = tf.reduce_sum(no_na_mask[...,0], axis=-1, keepdims=False)
            object_size = object_size * no_na_mask[...,0]
            norm = tf.math.divide_no_nan(tf.reduce_sum(object_size, axis=-1, keepdims=False), n_obj) * im_scale #(B, 1, 1, C)
            d = d * norm
        return d
    return loss
def _get_spatial_kernels(Y, X, two_channel_axes=True):
    Y, X = tf.meshgrid(tf.range(Y, dtype = tf.float32), tf.range(X, dtype = np.float32), indexing = 'ij')
    if two_channel_axes:
        return Y[tf.newaxis][..., tf.newaxis, tf.newaxis], X[tf.newaxis][..., tf.newaxis, tf.newaxis]
    else:
        return Y[tf.newaxis][..., tf.newaxis], X[tf.newaxis][..., tf.newaxis]

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
