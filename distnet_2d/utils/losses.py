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
    def __init__(self, focal_weight = 2.0, temperature: float = 0.0, pseudo_huber: float = 0.0, label_smoothing: float = 0, **kwargs):
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

            temperature: Tempering parameter (0 ≤ t < 1). Replaces log(p) with the tempered
                   logarithm log_t(p) = (p^t - 1) / t, whose gradient p^(t-1) decays
                   instead of blowing up like 1/p. This bounds the per-pixel loss on
                   confident-wrong / hard pixels (loss ≤ 1/t for p→0), so a few ambiguous
                   or mislabeled examples can no longer emit the huge gradients that
                   destabilize mixed-precision training. Higher t = tighter bound (1/t → 1
                   as t → 1) and more robustness to label noise, but may slow learning of
                   confident decisions.
                   t=0.0 → standard cross entropy (log, unbounded gradient)
                   t=0.1 → mild bounding (loss capped at -10 for p→0)
                   t=0.5 → strong bounding (loss capped at -2, very stable, may slow learning)

            pseudo_huber: Pseudo-Huber strength c (0 ≤ c ≤ 1), in units of chance level 1/K.
                   Internally floors the log by a = c/K (K = #classes): log(p) → log(p + c/K),
                   the smooth pseudo-Huber analog of cross entropy — CE for p ≫ c/K, linear
                   (MAE / L1, constant gradient) for p ≪ c/K, so a confident-wrong term's
                   gradient is bounded by 1/a = K/c instead of CE's unbounded 1/p. c=0 → CE.

                   Why c and not the raw floor a: 1/K (chance) is the natural probability unit
                   — the analog of the pixel unit for a regression pseudo-Huber delta — so c
                   is K-independent and reads directly as "the fraction of chance below which
                   an example is treated as an outlier". At chance p=1/K the CE gradient is K
                   and the floor caps it at K/c, so c<1 caps only worse-than-chance preds.
                     c ≈ 0.1–0.3 → cap only well-below-chance preds (clear mislabels), CE
                                   elsewhere — recommended
                     c = 1        → transition exactly at chance (more robust)
                   Equivalent readings: max gradient = K/c (lower tail; + γ|log(c/K)| with
                   focal γ), max loss ≈ log(K/c).

                   Applied to the whole probability vector, so WITH label_smoothing>0 the
                   smoothed wrong-class terms (which diverge as p→0 under over-confidence) are
                   floored too — capping the OVER-confident end as well; without smoothing only
                   the confident-wrong (true-class) tail is capped. Keep c ≲ ε (the smoothing
                   target is ε/K vs the floor c/K) to preserve smoothing's anti-overconfidence
                   push if overflow is the concern. Alternative robustifier to `temperature`
                   (temperature bounds the loss *value*, pseudo_huber the *gradient*); use one.

            label_smoothing: Smoothing parameter (0 ≤ ε < 1). y_smooth = y*(1-ε) + ε/K
                            (K=num_classes). Prevents over-confidence, improves calibration,
                            and bounds logit growth (so it also fights the fp16 logit-overflow).

                            Choosing ε — natural anchor: ε ≈ label noise / ambiguity rate ρ
                            (don't demand more confidence than the labels deserve; analog of
                            setting a regression Huber delta at the noise level). Three reads
                            of the same ε:
                              - confidence ceiling: optimum p_true ≈ 1-ε  (K-independent)
                              - logit bound: converged logit gap ≈ ln(K/ε); to keep gap ≤ Z
                                use ε ≥ K·e^{-Z} (logarithmic -> even tiny ε bounds it)
                              - per-wrong-class floor: ε/K
                            By purpose:
                              - overflow / logit control: tiny ε suffices (ε≈0.01 -> gap≈5.7);
                                use the smallest that bounds, to perturb labels least
                              - calibration / noise robustness: ε ≈ ρ (typ. 0.01-0.1)
                              - small K (e.g. 3): ImageNet's 0.1 (tuned for K=1000) is strong;
                                prefer 0.01-0.05
                            Match to the logit soft-cap so they cooperate: the soft-cap bounds
                            each logit to ~±c_softcap, so the max gap is ~2·c_softcap; the
                            smoothed optimum gap is ln(K/ε), so it stays inside the cap when
                            ε ≳ K·e^{-2·c_softcap} (c_softcap=4,K=3 -> ε ≳ 1e-3; ε≈0.01 sits
                            comfortably inside). Much smaller ε -> smoothing wants a gap beyond
                            2·c_softcap and tanh fights it; much larger -> cap rarely engages.
                            Practical range ~[0.005, 0.2]; don't stack heavy smoothing with
                            heavy focal/temperature/pseudo_huber (all damp confidence ->
                            under-training); too large -> genuinely under-confident (≤ 1-ε).
        """
        if focal_weight is None or (isinstance(focal_weight, (float, int)) and focal_weight == 0):
            self.focal_weight = None
        else:
            self.focal_weight = np.atleast_1d(np.array(focal_weight, dtype=np.float32))
        self.temperature = float(temperature)
        self.pseudo_huber = float(pseudo_huber)
        self.label_smoothing = float(label_smoothing)
        print(f"Cat. Loss: focal weight: {self.focal_weight} label smoothing: {self.label_smoothing} temperature: {self.temperature} pseudo_huber: {self.pseudo_huber}")
        assert self.focal_weight is None or np.all(self.focal_weight >= 0), f"gamma must be >=0, got {focal_weight}"
        assert 1 > self.temperature >= 0, f"temperature must be >=0 and <1, got {temperature}"
        assert 0 <= self.pseudo_huber <= 1, f"pseudo_huber (c, in units of chance 1/K) must be in [0,1], got {pseudo_huber}"
        assert 0 <= label_smoothing < 1, f"label_smoothing must be in [0,1), got {label_smoothing}"
        # Tempered log: log_t(p) = (p^t - 1)/t, gradient p^(t-1). t=0 is the 0/0 limit
        # = standard log (handled separately below). pseudo_huber floors the log by a=c/K
        # (c=self.pseudo_huber, K=#classes): log(p+a) -> gradient bounded by 1/a=K/c.

        super().__init__(**kwargs)

    def _tempered_log(self, p, floor=0.):
        # Pseudo-Huber floor a (=c/K): log(p+a) stays CE for p>>a, linear (bounded grad 1/a) for p<<a.
        if self.pseudo_huber > 0.:
            p = p + tf.cast(floor, p.dtype)
        # t=0: standard natural log (unbounded gradient). 0<t<1: bounded tempered log (loss <= 1/t).
        if self.temperature == 0.:
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

        # Pseudo-Huber floor a = c/K (K = #classes), in units of chance level; 0 if disabled
        floor = self.pseudo_huber / tf.cast(tf.shape(y_pred)[-1], y_pred.dtype) if self.pseudo_huber > 0 else 0.
        # Combined loss (tempered log / pseudo-Huber floor bound the per-pixel loss / gradient)
        loss = - focal_weight * y_true * self._tempered_log(y_pred, floor)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'temperature': self.temperature,
            'pseudo_huber': self.pseudo_huber,
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
