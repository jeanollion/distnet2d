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
    def __init__(self, focal_weight = 2.0, temperature: float = 0.0, label_smoothing: float = 0, **kwargs):
        """
        Tempered Focal Cross-Entropy with Label Smoothing for multi-class classification.
        Combines hard example mining (focal), gradient bounding (temperature), and
        regularization (label smoothing).

        Args:
            focal_weight: Focusing parameter (γ ≥ 0). Controls hard example emphasis. can be a list / tuple -> one value for each class
                   γ=0.0 → tempered CE (no focal effect)
                   γ=1.0 → mild focus on hard examples
                   γ=2.0 → standard focal (recommended start)
                   γ=5.0 → extreme focus (for very imbalanced data)

            temperature: Tempering parameter (1 > t ≥ 0). Replaces log(p) with the tempered
                   logarithm log_t(p) = (p^(t) - 1) / t, whose gradient is
                   p^(t-1) instead of 1/p. This bounds the loss (and gradient) on
                   confident-wrong / hard pixels, so a few ambiguous or mislabeled
                   examples can no longer emit the huge gradients that destabilize
                   mixed-precision training. Higher t = tighter bound (loss → 1/t,
                   i.e. → 1 as t → ∞) and more robustness to label noise, but may slow
                   learning of confident decisions.
                   t=0.0 → standard cross entropy (log, unbounded gradient)
                   t=0.1 → moderate bounding (loss capped at 10 for p→0)
                   t=0.5+ → strong bounding (very stable, may slow learning)

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
        self.temperature = float(temperature)
        self.label_smoothing = float(label_smoothing)
        print(f"Cat. Loss: focal weight: {self.focal_weight} label smoothing: {self.label_smoothing} temperature: {self.temperature}")
        assert self.focal_weight is None or np.all(self.focal_weight >= 0), f"gamma must be >=0, got {focal_weight}"
        assert 1 > self.temperature >= 0, f"temperature must be >=0 and <1, got {temperature}"
        assert 0 <= label_smoothing < 1, f"label_smoothing must be in [0,1), got {label_smoothing}"
        # Exponent of the tempered log: log_t(p) = (p^temperature - 1)/_temper_exp, with
        # gradient p^(_temper_exp-1) = p^(-1/t). t=0 -> standard log (handled separately).

        super().__init__(**kwargs)

    def _tempered_log(self, p):
        # t=1: standard natural log (unbounded). t>1: bounded tempered log.
        if self.temperature == 1.:
            return tf.math.log(p)
        e = tf.cast(self.temperature, p.dtype)
        return (tf.pow(p, e) - 1.) / e

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
        # Note: With label smoothing, focal effect is slightly reduced since targets are no longer pure 0/1
        if self.focal_weight is not None:
            if len(self.focal_weight) == 1:
                focal_weight = tf.pow(1. - y_pred, tf.constant(self.focal_weight[0], dtype=y_pred.dtype))
            else: # per class gamma
                weight_tensor = tf.constant(self.focal_weight, dtype=y_pred.dtype)
                weight_tensor = tf.reshape(weight_tensor, [1] * (len(y_pred.shape) - 1) + [-1])
                focal_weight = tf.pow(1. - y_pred, weight_tensor)
        else:
            focal_weight = tf.cast(1, y_true.dtype)

        # Combined loss (tempered log bounds the per-pixel loss/gradient when t>1)
        loss = - focal_weight * y_true * self._tempered_log(y_pred)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'focal_weight': list(self.focal_weight) if self.focal_weight is not None else None,
            'label_smoothing': self.label_smoothing
        })
        return config


def _apply_der_mask(tensor, der_mask):
    return tf.where(der_mask, tensor, 0) if der_mask is not None else tensor


def compute_loss_derivatives(true, pred, loss_fun, true_dy=None, true_dx=None, pred_dy=None, pred_dx=None, pred_lap=None, true_dz=None, pred_dz=None, mask=None, der_mask=None, derivative_loss: bool = False, laplacian_loss: bool = False, weight_map=None):
    loss = loss_fun(true, tf.where(mask, pred, 0) if mask is not None else pred)
    if weight_map is not None:
        loss = loss * weight_map
    has_pred_ders = pred_dy is not None or pred_dx is not None or pred_dz is not None
    if derivative_loss or laplacian_loss or has_pred_ders or pred_lap is not None:
        if der_mask is None:
            der_mask = mask
        tridim = len(true.shape) == 5
        # compute true derivatives
        if tridim:
            if true_dz is None:
                true_dz = der.der(true, 1)
            if true_dy is None:
                true_dy = der.der(true, 2)
            if true_dx is None:
                true_dx = der.der(true, 3)
        else:
            if true_dy is None:
                true_dy = der.der(true, 1)
            if true_dx is None:
                true_dx = der.der(true, 2)
        if derivative_loss or laplacian_loss:
            if tridim:
                pred_dz_comp = der.der(pred, 1)
                pred_dy_comp = der.der(pred, 2)
                pred_dx_comp = der.der(pred, 3)
            else:
                pred_dy_comp = der.der(pred, 1)
                pred_dx_comp = der.der(pred, 2)
        if laplacian_loss or pred_lap is not None:
            if tridim:
                true_lap = der.laplacian(derivatives=[true_dz, true_dy, true_dx])
            else:
                true_lap = der.laplacian(derivatives=[true_dy, true_dx])
        if pred_lap is not None:
            der_loss = loss_fun(true_lap, _apply_der_mask(pred_lap, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
        if laplacian_loss:
            if tridim:
                lap_pred = der.laplacian(derivatives=[pred_dz_comp, pred_dy_comp, pred_dx_comp])
            else:
                lap_pred = der.laplacian(derivatives=[pred_dy_comp, pred_dx_comp])
            der_loss = loss_fun(true_lap, _apply_der_mask(lap_pred, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
        # explicit predicted derivatives (from network heads)
        if pred_dz is not None:
            der_loss = loss_fun(true_dz, _apply_der_mask(pred_dz, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
        if pred_dy is not None:
            der_loss = loss_fun(true_dy, _apply_der_mask(pred_dy, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
        if pred_dx is not None:
            der_loss = loss_fun(true_dx, _apply_der_mask(pred_dx, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
        # derivative loss (from computed derivatives of prediction)
        if derivative_loss:
            if tridim:
                der_loss = loss_fun(true_dz, _apply_der_mask(pred_dz_comp, der_mask))
                loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
            der_loss = loss_fun(true_dy, _apply_der_mask(pred_dy_comp, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
            der_loss = loss_fun(true_dx, _apply_der_mask(pred_dx_comp, der_mask))
            loss = loss + (der_loss * weight_map if weight_map is not None else der_loss)
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
