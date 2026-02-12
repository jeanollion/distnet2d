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


class FocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, focal_weight = 2.0, label_smoothing: float = 0, **kwargs):
        """
        Tempered Focal Cross-Entropy with Label Smoothing for multi-class classification.
        Combines hard example mining (focal), and regularization (label smoothing).

        Args:
            focal_weight: Focusing parameter (γ ≥ 0). Controls hard example emphasis. can be a list / tuple -> one value for each class
                   γ=0.0 → tempered CE (no focal effect)
                   γ=1.0 → mild focus on hard examples
                   γ=2.0 → standard focal (recommended start)
                   γ=5.0 → extreme focus (for very imbalanced data)

            label_smoothing: Smoothing parameter (0 ≤ ε < 1). Regularization strength.
                            ε=0.0 → no smoothing (hard labels)
                            ε=0.1 → typical for ImageNet (recommended start)
                            ε=0.2 → stronger regularization
                            Effect: y_smooth = y * (1-ε) + ε/K where K=num_classes

                            Benefits:
                            - Prevents overconfidence (probabilities ≠ 0 or 1)
                            - Improves calibration (predicted probs match true frequencies)
                            - Acts as regularization (reduces overfitting)
                            - Better generalization on test data

                            When useful:
                            - Models prone to overconfidence
                            - Limited training data
                            - Noisy labels
                            - When calibration matters (e.g., medical, finance)

                            Trade-offs:
                            - May slightly hurt training accuracy
                            - Improves test accuracy & calibration
                            - Can conflict with focal loss (both modify targets)
        """
        if focal_weight is None or (isinstance(focal_weight, (float, int)) and focal_weight == 0):
            self.focal_weight = None
        else:
            self.focal_weight = np.atleast_1d(np.array(focal_weight, dtype=np.float32))
        self.label_smoothing = float(label_smoothing)

        assert focal_weight is None or np.all(focal_weight >= 0), f"gamma must be >=0, got {focal_weight}"
        assert 0 <= label_smoothing < 1, f"label_smoothing must be in [0,1), got {label_smoothing}"

        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Args:
            y_true: One-hot encoded labels, shape (batch_size, (Y, X), num_classes)
            y_pred: Predicted probabilities, shape (batch_size, (Y, X), num_classes)
        """
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Apply label smoothing: y_smooth = y * (1-ε) + ε/K
        if self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], y_true.dtype)
            y_true = y_true * (1. - self.label_smoothing) + self.label_smoothing / num_classes

        # Focal weight: (1 - p)^gamma
        # Note: With label smoothing, focal effect is slightly reduced
        # since targets are no longer pure 0/1
        if self.focal_weight is not None:
            if len(self.focal_weight) == 1:
                focal_weight = tf.pow(1. - y_pred, tf.constant(self.focal_weight[0], dtype=y_pred.dtype))
            else: # per class gamma
                weight_tensor = tf.constant(self.focal_weight, dtype=y_pred.dtype)
                weight_tensor = tf.reshape(weight_tensor, [1] * (len(y_pred.shape) - 1) + [-1])
                focal_weight = tf.pow(1. - y_pred, weight_tensor)
        else:
            focal_weight = tf.cast(1, y_true.dtype)

        # Combined loss
        loss = - focal_weight * y_true * tf.math.log(y_pred)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'focal_weight': list(self.focal_weight) if self.focal_weight is not None else None,
            'label_smoothing': self.label_smoothing
        })
        return config


def compute_loss_derivatives(true, pred, loss_fun, true_dy=None, true_dx=None, pred_dy=None, pred_dx=None, pred_lap=None, mask=None, der_mask=None, derivative_loss: bool = False, laplacian_loss: bool = False, weight_map=None):
    loss = loss_fun(true, tf.where(mask, pred, 0) if mask is not None else pred)
    if weight_map is not None:
        loss = loss * weight_map
    #print(f"compute loss with mask: {mask is not None} interior: {mask_interior is not None} der: {derivative_loss} grad: {gradient_loss} lap: {laplacian_loss} pred lap: {y_pred_lap is not None} pred dy: {y_pred_dy is not None} pred dx: {y_pred_dx is not None}", flush=True)
    if derivative_loss or laplacian_loss or pred_dy is not None or pred_dx is not None or pred_lap is not None:
        if der_mask is None:
            der_mask = mask
        if true_dy is None:
            true_dy = der.der_2d(true, 1)
        if true_dx is None:
            true_dx = der.der_2d(true, 2)
        if derivative_loss or laplacian_loss:
            dy_pred, dx_pred = der.der_2d(pred, 1), der.der_2d(pred, 2)
        if laplacian_loss or pred_lap is not None:
            true_lap = der.laplacian_2d(None, true_dy, true_dx)
        if pred_lap is not None:
            pred_lap = tf.where(der_mask, pred_lap, 0) if der_mask is not None else pred_lap
            loss = loss + loss_fun(true_lap, pred_lap)
        if laplacian_loss:
            lap_pred = der.laplacian_2d(None, dy_pred, dx_pred)
            lap_pred = tf.where(der_mask, lap_pred, 0) if der_mask is not None else lap_pred
            loss = loss + loss_fun(true_lap, lap_pred)
        if pred_dy is not None:
            pred_dy = tf.where(der_mask, pred_dy, 0) if der_mask is not None else pred_dy
            loss = loss + loss_fun(true_dy, pred_dy)
        if pred_dx is not None:
            pred_dx = tf.where(der_mask, pred_dx, 0) if der_mask is not None else pred_dx
            loss = loss + loss_fun(true_dx, pred_dx)
        if derivative_loss:
            dy_pred = tf.where(der_mask, dy_pred, 0) if der_mask is not None else dy_pred
            dx_pred = tf.where(der_mask, dx_pred, 0) if der_mask is not None else dx_pred
            loss = loss + loss_fun(true_dy, dy_pred) + loss_fun(true_dx, dx_pred)
    return loss


def weighted_loss_by_category(original_loss_func, weight_list, sparse=True, remove_background=False, dtype='float32'):
    if isinstance(weight_list, (list, tuple)):
        weight_list = np.array(weight_list, dtype=dtype)
    n_classes = tf.shape(weight_list)[0]
    def loss_func(y_true, y_pred, sample_weight=None):
        weights = tf.cast(weight_list, dtype)
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

        class_weights = tf.reduce_sum(class_weights * weights, axis=-1, keepdims=False) # multiply with broadcast
        if sample_weight is not None:
            class_weights = sample_weight * class_weights
        return original_loss_func(y_true, y_pred, sample_weight=class_weights)
    return loss_func

def balanced_category_loss(original_loss_func, n_classes, max_class_frequency=10, sparse=True, remove_background=False, dtype='float32'):
    max_class_frequency = np.array([max_class_frequency]).astype(dtype)
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
        weight_list = tf.math.minimum(max_class_frequency[0], weight_list)

        weight_list = tf.cast(weight_list, dtype=dtype)
        #print(f"class weights: {weight_list.numpy()}")
        class_weights = tf.reduce_sum(class_weights * weight_list, axis=-1, keepdims=False) # multiply with broadcast
        if sample_weight is not None:
            class_weights = sample_weight * class_weights
        return original_loss_func(y_true, y_pred, sample_weight=class_weights)
    return loss_func
