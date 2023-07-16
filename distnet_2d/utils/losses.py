import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def get_grad_weight_fun(weight):
    @tf.custom_gradient
    def wgrad(x):
        def grad(dy):
            if isinstance(dy, tuple): #and len(dy)>1
                #print(f"gradient is tuple of length: {len(dy)}")
                return (y * weight for y in dy)
            elif isinstance(dy, list):
                #print(f"gradient is list of length: {len(dy)}")
                return [y * weight for y in dy]
            else:
                return dy * weight
        return x, grad
    return wgrad

class PseudoHuber(tf.keras.losses.Loss):
    def __init__(self, delta:float = 1., **kwargs):
        self.delta = float(delta)
        self.delta_sq = self.delta * self.delta
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return tf.multiply(self.delta_sq, tf.sqrt(1. + tf.square((y_true - y_pred)/self.delta)) - 1.)

def weighted_loss_by_category(original_loss_func, weight_list, axis=-1, sparse=True, remove_background=False, dtype='float32'):
    weight_list = np.array(weight_list).astype("float32")
    # normalize weights:
    n_classes = weight_list.shape[0]
    weight_list = n_classes * weight_list / np.sum(weight_list)
    weight_list = tf.cast(weight_list, dtype=dtype)
    def loss_func(y_true, y_pred, sample_weight=None):
        if sparse:
            class_weights = tf.squeeze(y_true, axis=-1)
            if not class_weights.dtype.is_integer:
                class_weights = tf.cast(class_weights, tf.int32)
            class_weights = tf.one_hot(class_weights, n_classes+(1 if remove_background else 0), dtype=dtype)
            if remove_background:
                class_weights = class_weights[...,1:]
            y_true = class_weights
        else:
            if remove_background:
                y_true = y_true[...,1:]
            class_weights = tf.cast(y_true, dtype=dtype)

        class_weights = tf.reduce_sum(class_weights * weight_list, axis=-1, keepdims=False) # multiply with broadcast
        if sample_weight is not None:
            class_weights = sample_weight * class_weights
        return original_loss_func(y_true, y_pred, sample_weight=class_weights)
    return loss_func

def balanced_category_loss(original_loss_func, n_classes, min_class_frequency=1./10, max_class_frequency=10, axis=-1, sparse=True, remove_background=False, dtype='float32'):
    weight_limits = np.array([min_class_frequency, max_class_frequency]).astype(dtype)
    def loss_func(y_true, y_pred, sample_weight=None):
        if sparse:
            class_weights = tf.squeeze(y_true, axis=-1)
            if not class_weights.dtype.is_integer:
                class_weights = tf.cast(class_weights, tf.int32)
            class_weights = tf.one_hot(class_weights, n_classes+(1 if remove_background else 0), dtype=dtype)
            if remove_background:
                class_weights = class_weights[...,1:]
            y_true = class_weights
        else:
            if remove_background:
                y_true = y_true[...,1:]
            class_weights = tf.cast(y_true, dtype=dtype)

        class_count = tf.math.count_nonzero(class_weights, axis=tf.range(tf.rank(class_weights)-1), dtype=tf.float32)
        count = tf.reduce_sum(class_count)
        weight_list = tf.math.divide_no_nan(count, class_count)
        weight_list = tf.math.divide_no_nan(weight_list, tf.cast(n_classes, dtype=tf.float32)) # divide by class number so that balanced frequency of each class corresponds to the same frequency of 1/n_classes
        weight_list = tf.math.minimum(weight_limits[1], tf.math.maximum(weight_limits[0], weight_list))

        weight_list = tf.cast(weight_list, dtype=dtype)
        #print(f"class weights: {weight_list.numpy()}")
        class_weights = tf.reduce_sum(class_weights * weight_list, axis=-1, keepdims=False) # multiply with broadcast
        if sample_weight is not None:
            class_weights = sample_weight * class_weights
        return original_loss_func(y_true, y_pred, sample_weight=class_weights)
    return loss_func
