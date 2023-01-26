import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError

def get_soft_argmax_2d_fun(beta=1e2, two_channel_axes=True):
    @tf.function
    def sam(x):
        Y, X = _get_spatial_kernels(*x.shape.as_list()[1:3], two_channel_axes)
        shape = tf.shape(x)
        count = tf.expand_dims(tf.math.count_nonzero(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, T, C, 1)
        x = tf.reshape(x, tf.concat([shape[:1], [-1], shape[-1:]], 0))
        x = tf.nn.softmax(x * beta, axis = 1)
        x = tf.reshape(x, shape)
        argmax_y = tf.reduce_sum(x*Y, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        argmax_x = tf.reduce_sum(x*X, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        argmax = tf.stack([argmax_y, argmax_x], -1) # (B, 1, 1, T, C, 2)
        return tf.where(count==0, tf.constant(float('NaN')), argmax) # when no values should return nan
    return sam

def get_weighted_mean_2d_fun(two_channel_axes=True):
    @tf.function
    def wm(x):
        shape = tf.shape(x)
        Y, X = _get_spatial_kernels(*x.shape.as_list()[1:3], two_channel_axes)
        wsum_y = tf.reduce_sum(x * Y, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        wsum_x = tf.reduce_sum(x * X, axis=[1, 2], keepdims=True) # (B, 1, 1, T, C)
        wsum = tf.stack([wsum_y, wsum_x], -1) # (B, 1, 1, T, C, 2)
        sum = tf.expand_dims(tf.reduce_sum(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, T, C, 1)
        return tf.math.divide(wsum, sum) # when no values should return nan # (B, 1, 1, T, C, 2)
    return wm

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