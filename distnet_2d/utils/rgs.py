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

class RelativeGradientScaler():
    def __init__(self, parameters, ref_key, momentum=0.98):
        self.momentum = momentum
        self.parameters = {k:list(zip(*p))[0] for k, p in parameters.items()}
        self.parameter_indices = {k:list(zip(*p))[1] for k, p in parameters.items()}
        self.ref_key = ref_key
        self.grad_norms = None
        self.key_index = {k:i for i, k in enumerate(parameters.keys())}
        self.norm_accumulation = [
            tf.Variable(0., trainable=False, name=f"norm_{k}",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=tf.float32) for k in self.key_index.keys()
        ]
        self.initialized = [
            tf.Variable(False, trainable=False, name=f"init_{k}",
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
            dtype=tf.bool) for k in self.key_index.keys()
        ]
        self.cur_momentum = tf.Variable(0., trainable=False, name="cur_momentum",
        synchronization=tf.VariableSynchronization.ON_READ,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
        dtype=tf.float32)

    def update(self, losses, tape, grad_eps = 1e-6):
        for k, l in losses.items():
            with tape.stop_recording():
                grads = tape.gradient(l, self.parameters[k])
                if not isinstance(grads, (list, tuple)):
                    grads = [grads]
            norms = [tf.stop_gradient(_unitwise_norm(g)/tf.math.maximum(_unitwise_norm(p), grad_eps)) for p,g in zip(self.parameters[k], grads) if g is not None]
            print(f"norm shape {[n.shape for n in norms]}")
            if len(norms)>0:
                norm = tf.math.reduce_mean(tf.stack(norms), keepdims = False)
                self._update_norm(norm, k)
        self.cur_momentum.assign(tf.math.minimum(0.5 + self.cur_momentum * 0.5, self.momentum)) # increase momentum to target value
        print(f"momentum: {self.cur_momentum}")

    def _update_norm(self, norm, key, scaled_gradients = False):
        # print(f"update norm: {norms}")
        i = self.key_index[key]
        def init():
            self.initialized[i].assign(True)
            return norm
        def update():
            if scaled_gradients:
                n  = norm * self.norm_accumulation[i] # gradient was computed after applying the previous weight -> compensate
            else:
                n = norm
            return self.norm_accumulation[i] * self.cur_momentum  + n * (1. - self.cur_momentum )
        #print(f"norm: {key} = {self.norm_accumulation[i]} init: {self.initialized[i]}")
        self.norm_accumulation[i].assign(tf.cond(self.initialized[i], update, init))
        print(f"norm: {key} = {self.norm_accumulation[i]} from: {norm}")

    def scale_gradients(self, losses):
        ref_idx = self.key_index[self.ref_key]
        for k, l in losses.items():
            if k!=self.ref_key:
                i = self.key_index[k]
                print(f"scale {k} = {self.norm_accumulation[ref_idx]/self.norm_accumulation[i]}")
                g_weight = get_grad_weight_fun(self.norm_accumulation[ref_idx]/self.norm_accumulation[i])
                losses[k] = g_weight(l)
