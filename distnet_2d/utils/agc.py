#
""" An implementation of Adaptive Gradient Clipping
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
Code references:
  * Official JAX implementation (paper authors): https://github.com/deepmind/deepmind-research/tree/master/nfnets
  * Ross Wightman's implementation https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/agc.py
  * This is the implementation from André Pedersen and David Bouget: https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/agc.py
"""
import tensorflow as tf


# implementation from: https://github.com/sayakpaul/Adaptive-Gradient-Clipping/blob/main/agc.py
def compute_norm(x, axis, keepdims):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(x), axis=axis, keepdims=keepdims))

def unitwise_norm(x):
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
    return compute_norm(x, axis, keepdims)


def adaptive_clip_grad(parameters, gradients, clip_factor=0.01, eps=1e-3, grad_eps=1e-3, grad_scale=1., exclude_keywords=None, monitor:bool = False):
    new_grads = []
    if monitor:
        total_clipped = 0.
        total_grads = 0.
    for (params, grads) in zip(parameters, gradients):
        if params is None or grads is None:
            new_grads.append(grads)
        elif exclude_gradient(params.name, exclude_keywords):
            if grad_scale != 1.:
                grads = tf.math.multiply(grads, grad_scale)
            new_grads.append(grads)
        else:
            # Scale gradients first if needed
            if grad_scale != 1.:
                grads = tf.math.multiply(grads, grad_scale)
            grads = tf.where(
                tf.math.is_finite(grads),
                grads,
                tf.zeros_like(grads)
            )
            p_norm = unitwise_norm(params)
            max_norm = tf.math.maximum(p_norm, eps) * clip_factor
            grad_norm = unitwise_norm(grads)
            clipped_grad = grads * (max_norm / tf.math.maximum(grad_norm, grad_eps))
            to_clip = grad_norm > max_norm
            new_grad = tf.where(to_clip, clipped_grad, grads)
            new_grads.append(new_grad)
            if monitor:
                total_clipped += tf.reduce_sum(tf.cast(to_clip, tf.float32))
                total_grads += tf.cast(tf.size(to_clip), tf.float32)

    if monitor:
        clip_ratio = 100. * total_clipped / total_grads
        tf.print(f"AGC clipping ratio: {clip_ratio}%")
    return new_grads

def exclude_gradient(name, exclude_keywords):
    if exclude_keywords is None:
        return False
    if not isinstance(exclude_keywords, (tuple, list)):
        exclude_keywords = [exclude_keywords]
    for k in exclude_keywords:
        if k in name:
            return True
    return False
