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

# def _get_scale_shape(x):
#     shape = x.shape.as_list()
#     if len(shape) <= 1:  # Scalars and vectors
#         return (1,)
#     elif len(shape) == 2:  # Linear layers of shape IO or multihead linear
#         return (1, shape[-1])
#     elif len(shape) == 3:
#         return (1, 1, shape[-1])
#     elif len(shape) == 4:  # Conv kernels of shape HWIO
#         return (1, 1, 1, shape[-1])
#     elif len(shape) == 5:  # Conv kernels of shape HWDIO
#         return (1, 1, 1, 1, shape[-1])
#     else:
#         raise ValueError(f"Got a parameter with shape not in [1, 2, 4, 5]! {x}")

class AdaptativeGradientScaler():
    def __init__(self, clip_factor:float, unitwise:bool = True, eps:float=1e-3, grad_eps:float= 1e-6):
        self.eps=eps
        self.grad_eps = grad_eps
        self.clip_factor = clip_factor
        self.scale = None
        self.unitwise=unitwise

    def init_parameters(self, parameters:dict):
        self.parameters = parameters
        for k, params in parameters.items():
            if len(params)>1:
                for p in params[1:]:
                    assert params[0].shape.as_list() == p.shape.as_list(), "all parameters should have same shape for a given loss"
        self.key_index = {k:i for i, k in enumerate(parameters.keys())}
        self.scale = [
            tf.Variable(1., dtype = tf.float32, trainable=False, name=f"scale_{k}",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA) for k, p in parameters.items()
        ]

    def initialized(self):
        return self.scale is not None

    def _get_scale(self, params, grads):
        p_norm = _unitwise_norm(params)
        max_norm = tf.math.maximum(p_norm, self.eps) * self.clip_factor
        grad_norm = _unitwise_norm(grads)
        scale = tf.math.divide(max_norm, tf.math.maximum(grad_norm, self.grad_eps))
        if not self.unitwise:
            scale = tf.math.reduce_mean(scale)
        # print(f"all scales: {scale}")
        scale = tf.math.minimum(1., scale)
        #return scale
        return tf.math.reduce_mean(scale)

    def update(self, losses, tape, loss_weights=None):
        for k, l in losses.items():
            if loss_weights is not None and k in loss_weights and loss_weights[k]!=1:
                l = l * loss_weights[k]
            with tape.stop_recording():
                grads = tape.gradient(l, self.parameters[k])
                if not isinstance(grads, (list, tuple)):
                    grads = [grads]
            scales = [self._get_scale(p,g) for p,g in zip(self.parameters[k], grads) if g is not None]
            if len(scales)>0:
                scale = tf.math.reduce_mean(tf.stack(scales), axis = 0, keepdims = False)
                scale = tf.stop_gradient(scale) # TODO : necessary ?
                i = self.key_index[k]
                self.scale[i].assign(scale)
                print(f"loss: {k} scale: {scale}")

    def scale_gradients(self, losses):
        for k, l in losses.items():
            i = self.key_index[k]
            g_weight = get_grad_weight_fun(self.scale[i])
            losses[k] = g_weight(l)
