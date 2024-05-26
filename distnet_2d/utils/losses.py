import tensorflow as tf
import numpy as np
from ..utils import image_derivatives_tf as der


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


def compute_loss_derivatives(true, pred, loss_fun, true_dy=None, true_dx=None, pred_dy=None, pred_dx=None, pred_lap=None, mask=None, mask_interior=None, derivative_loss: bool = False, laplacian_loss: bool = False):
    loss = loss_fun(true, tf.where(mask, pred, 0) if mask is not None else pred)
    #print(f"compute loss with mask: {mask is not None} interior: {mask_interior is not None} der: {derivative_loss} grad: {gradient_loss} lap: {laplacian_loss} pred lap: {y_pred_lap is not None} pred dy: {y_pred_dy is not None} pred dx: {y_pred_dx is not None}", flush=True)
    if derivative_loss or laplacian_loss or pred_dy is not None or pred_dx is not None or pred_lap is not None:
        if mask_interior is None:
            mask_interior = mask
        if true_dy is None:
            true_dy = der.der_2d(true, 1)
        if true_dx is None:
            true_dx = der.der_2d(true, 2)
        if derivative_loss or laplacian_loss:
            dy_pred, dx_pred = der.der_2d(pred, 1), der.der_2d(pred, 2)
        if laplacian_loss or pred_lap is not None:
            true_lap = der.laplacian_2d(None, true_dy, true_dx)
        if pred_lap is not None:
            pred_lap = tf.where(mask_interior, pred_lap, 0) if mask_interior is not None else pred_lap
            loss = loss + loss_fun(true_lap, pred_lap)
        if laplacian_loss:
            lap_pred = der.laplacian_2d(None, dy_pred, dx_pred)
            lap_pred = tf.where(mask_interior, lap_pred, 0) if mask_interior is not None else lap_pred
            loss = loss + loss_fun(true_lap, lap_pred)
        if pred_dy is not None:
            pred_dy = tf.where(mask_interior, pred_dy, 0) if mask_interior is not None else pred_dy
            loss = loss + loss_fun(true_dy, pred_dy)
        if pred_dx is not None:
            pred_dx = tf.where(mask_interior, pred_dx, 0) if mask_interior is not None else pred_dx
            loss = loss + loss_fun(true_dx, pred_dx)
        if derivative_loss:
            dy_pred = tf.where(mask_interior, dy_pred, 0) if mask_interior is not None else dy_pred
            dx_pred = tf.where(mask_interior, dx_pred, 0) if mask_interior is not None else dx_pred
            loss = loss + loss_fun(true_dy, dy_pred) + loss_fun(true_dx, dx_pred)
    return loss


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
