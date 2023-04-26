# relative gradient scaling

import tensorflow as tf
from .losses import get_grad_weight_fun

# implementation from: https://github.com/sayakpaul/Adaptive-Gradient-Clipping/blob/main/agc.py
def _compute_norm(x, axis, keepdims):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x), axis=axis, keepdims=keepdims))

def _unitwise_norm(x):
    if isinstance(x, tf.IndexedSlices):
        x = tf.convert_to_tensor(x.values)
    shape = x.shape.as_list()
    if len(shape) <= 1:  # Scalars and vectors
        axis = None
        keepdims = False
    elif len(shape) in [2, 3]:  # Linear layers of shape IO or multihead linear
        axis = 0
        keepdims = True
    elif len(shape) == 4:  # Conv kernels of shape HWIO
        axis = [0, 1, 2]
        keepdims = True
    elif len(shape) == 5:  # Conv kernels of shape HWDIO
        axis = [0, 1, 2, 3]
        keepdims = True
    else:
        raise ValueError(f"Got a parameter with shape not in [1, 2, 4, 5]! {x}")
    return _compute_norm(x, axis, keepdims)

def _get_norm(gradients):
    return {k : tf.math.reduce_mean([_unitwise_norm(g) for g in grads], keepdims = False) for k, grads in gradients.items()}

class RelativeGradientScaler():
    def __init__(self, parameters, ref_key, momentum=0.98):
        self.momentum = momentum
        self.parameters = {k:list(zip(*p))[0] for k, p in parameters.items()}
        self.parameter_indices = {k:list(zip(*p))[1] for k, p in parameters.items()}
        self.ref_key = ref_key
        self.grad_norms = None

    def initialized(self):
        return self.grad_norms is not None

    def update(self, losses, tape):
        with tape.stop_recording():
            grads = {k:tf.stop_gradient(tape.gradient(losses[k], self.parameters[k])) for k in losses.keys()}
        self._update_norms(_get_norm(grads))

    # def update(self, gradients):
    #     grads = {k: [tf.stop_gradient(gradients[i]) for i in indices] for k, indices in self.parameter_indices.items()}
    #     self._update_norms(_get_norm(grads), scaled_gradients=True)

    def _update_norms(self, norms, scaled_gradients = False):
        # print(f"update norm: {norms}")
        if self.grad_norms is None:
            self.grad_norms = norms
        else:
            for k, w in norms.items():
                # print(f"update weight: {k} was: {self.grad_norms[k]}, new: {w} next: {self.grad_norms[k] * self.momentum + w * (1 - self.momentum)}")
                if scaled_gradients:
                    w  = w * self.relative_weights[k] # gradient was computed after applying the previous weight -> compensate
                self.grad_norms[k] = self.grad_norms[k] * self.momentum + w * (1 - self.momentum)

    def scale_gradients(self, losses):
        for k, l in losses.items():
            if k!=self.ref_key:
                g_weight = get_grad_weight_fun(self.grad_norms[self.ref_key]/self.grad_norms[k])
                losses[k] = g_weight(l)
