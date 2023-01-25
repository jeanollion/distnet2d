import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import MeanSquaredError

class SoftArgmax2D(Layer):
    def __init__(self, beta=1e2, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def get_config(self):
        config = super().get_config().copy()
        config.update({"beta": self.beta})
        return config

    def build(self, input_shape):
        _, Y, X, C = input_shape
        self.kernel = tf.constant(generate_kernel(Y, X, C), dtype = tf.float32)

    def call(self,x):
        B, Y, X, C = x.shape.as_list()
        shape = tf.shape(x)
        count = tf.expand_dims(tf.math.count_nonzero(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, C, 1)
        x = tf.reshape(x, (B, -1, C))
        x = tf.nn.softmax(x * self.beta, axis = 1)
        x = tf.reshape(x, (B, Y, X, C))
        argmax = tf.nn.depthwise_conv2d(x, self.kernel, strides= [1,1,1,1], padding='VALID')
        argmax = tf.reshape(argmax, tf.concat([shape[:1], [1, 1], shape[-1:], [2]], 0))
        return tf.where(count==0, tf.constant(float('NaN')), argmax) # when no values should return nan

    def compute_output_shape(self, input_shape):  # (B, Y, X, C)
        return (input_shape[0], 1, 1, input_shape[3], 2)  # (B, 1, 1, C, 2)

def get_soft_argmax_2d_fun(beta=1e2):
    @tf.function
    def sam(x):
        B, Y, X, C = x.shape.as_list()
        shape = tf.shape(x)
        kernel = tf.constant(generate_kernel(Y, X, C), dtype = tf.float32)
        count = tf.expand_dims(tf.math.count_nonzero(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, C, 1)
        x = tf.reshape(x, tf.concat([shape[:1], [-1], shape[-1:]], 0))
        x = tf.nn.softmax(x * beta, axis = 1)
        x = tf.reshape(x, shape)
        argmax = tf.nn.depthwise_conv2d(x, kernel, strides= [1,1,1,1], padding='VALID')
        argmax = tf.reshape(argmax, tf.concat([shape[:1], [1, 1], shape[-1:], [2]], 0))
        return tf.where(count==0, tf.constant(float('NaN')), argmax) # when no values should return nan
    return sam

class WeightedMean2D(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, Y, X, C = input_shape
        self.kernel = tf.constant(generate_kernel(Y, X, C), dtype = tf.float32)

    def call(self,x):
        _, Y, X, C = x.shape.as_list()
        shape = tf.shape(x)
        wsum = tf.nn.depthwise_conv2d(x, self.kernel, strides= [1,1,1,1], padding='VALID')
        wsum = tf.reshape(wsum, tf.concat([shape[:1], [1, 1], shape[-1:], [2]], 0))
        sum = tf.expand_dims(tf.reduce_sum(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, C, 1)
        return tf.math.divide(wsum, sum)

    def compute_output_shape(self, input_shape): # (B, Y, X, C)
        return (input_shape[0], 1, 1, input_shape[-1], 2) # (B, 1, 1, C, 2)


def get_weighted_mean_2d_fun():
    #@tf.function ??
    def wm(x):
        _, Y, X, C = x.shape.as_list() # HOW TO DO THIS IN EAGER MODE ? GENERATE KERNEL WITH TF ?
        shape = tf.shape(x)
        kernel = tf.constant(generate_kernel(Y, X, C), dtype = tf.float32)
        wsum = tf.nn.depthwise_conv2d(x, kernel, strides= [1,1,1,1], padding='VALID')
        wsum = tf.reshape(wsum, tf.concat([shape[:1], [1, 1], shape[-1:], [2]], 0))
        sum = tf.expand_dims(tf.reduce_sum(x, axis=[1, 2], keepdims = True), -1) # (B, 1, 1, C, 1)
        return tf.math.divide(wsum, sum) # when no values should return nan
    return wm

class GaussianSpread(Layer): # convert a coordinate to a gaussian distribution centered at the coordinate
    def __init__(self, sigma, sizeY=None, sizeX=None, kernel=None, objectwise=True, **kwargs):
        super().__init__(**kwargs)
        self.sigma_sq = sigma*sigma
        self.kernel=kernel
        self.objectwise=objectwise
        if kernel is not None:
            kshape = kernel.get_shape().as_list()
            if self.objectwise:
                assert tf.rank(kernel) == 6
            self.sizeY = kshape[1]
            self.sizeX = kshape[2]
        else:
            assert sizeY is not None and sizeX is not None, "sizeY and sizeX must be provided when kernel is not"
            self.sizeY = sizeY
            self.sizeX = sizeX

    def get_config(self):
        config = super().get_config().copy()
        config.update({"sigma_sq": self.sigma_sq, "sizeY":self.sizeY, "sizeX":self.sizeX, "objectwise":self.objectwise})
        return config

    def build(self, input_shape): # (B, 1, 1, C, 2) or (B, 1, 1, C, N, 2)
        assert input_shape[-1]==2, "only 2D coords supported"
        assert input_shape[1]==1 and input_shape[2]==1,"only points supported"
        # if self.objectwise:
        #     assert tf.size(input_shape) == 6
        if self.kernel is None:
            ker = generate_kernel(self.sizeY, self.sizeX, O=1 if self.objectwise else 0) # (sizeY, sizeX, 1, 2) or # (sizeY, sizeX, 1, 1, 2)
            ker = np.expand_dims(ker, 0) # (1, sizeY, sizeX, 1, 2) or  (1, sizeY, sizeX, 1, 1, 2)
            self.kernel = tf.constant(ker, dtype=tf.float32)

    def call(self,x): # input shape: (B, 1, 1, C, 2) or (B, 1, 1, C, N, 2)
        d = tf.math.reduce_sum(tf.math.square(x - self.kernel), axis=-1, keepdims=False) #(B, Y, X, C) or (B, Y, X, C, N)
        mul = -1./self.sigma_sq
        exp = tf.math.exp(d * mul)
        exp = tf.where(tf.math.is_nan(exp), tf.constant(0, dtype=tf.float32), exp)
        if self.objectwise:
            exp = tf.reduce_sum(exp, -1) #(B, Y, X, C)
        return exp

    def compute_output_shape(self, input_shape): # (B, Y, X, C)
        if self.objectwise:
            return (input_shape[0], self.sizeY, self.sizeX, input_shape[-3])
        else:
            return (input_shape[0], self.sizeY, self.sizeX, input_shape[-2])

def generate_kernel(sizeY, sizeX, C=1, O=0):
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
