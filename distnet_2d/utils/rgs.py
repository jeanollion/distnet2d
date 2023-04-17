# relative gradient scaling

import tensorflow as tf


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

def _get_rgs(gradients, ref):
    grad_norm = {k : tf.math.reduce_mean([_unitwise_norm(g) for g in grads], keepdims = False) for k, grads in gradients.items()}
    return { k : s / grad_norm[ref] for k,s in grad_norm.items() }

def scale_losses(losses, parameters, ref_key, tape):
    grads = {k:tf.stop_gradient(tape.gradient(losses[k], parameters[k])) for k in losses.keys()}
    scales = _get_rgs(grads, ref_key)
    print(f"relative loss scale: {scales}")
    return {k : losses[k] / scales[k] for k in losses.keys()}
