import math

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from collections import defaultdict

from distnet_2d.model.layers import HideVariableWrapper


class ClassWeightScheduler(Callback):
    def __init__(self, attribute_name:str, n_epochs:int, power_law:float=1., verbose:bool=True):
        super().__init__()
        self.n_epochs = float(n_epochs)
        self.power_law = float(power_law)
        self.attribute_name = attribute_name
        self.initial_weights = None
        self.final_weights = None
        self.verbose=verbose

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.initial_weights = tf.identity(self._get())
        self.final_weights = tf.ones_like(self.initial_weights)

    def _get(self):
        var = getattr(self.model, self.attribute_name)
        if isinstance(var, HideVariableWrapper):
            return var.value
        else:
            return var

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        alpha = tf.math.pow( float(epoch) / self.n_epochs , self.power_law )
        cur_weights = self.initial_weights * (1 - alpha) + self.final_weights * alpha # linear interpolation
        if self.verbose and epoch%100==0:
            print(f"Epoch: {epoch} weights for {self.attribute_name}: {cur_weights}", flush=True)
        self._get().assign( cur_weights )


class GradientMonitorCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_steps):
        super().__init__()
        self.n_steps = float(n_steps)
        self.epoch_stats = defaultdict(lambda: {
            'max_abs': tf.Variable(-float('inf'), dtype=tf.float32),
            'max_norm': tf.Variable(-float('inf'), dtype=tf.float32),
            'min_norm': tf.Variable(float('inf'), dtype=tf.float32),
            'mean_norm': tf.Variable(0.0, dtype=tf.float32)
        })

    def on_train_begin(self, logs=None):
        self.model.log_gradients = self.accumulate_gradients

    def accumulate_gradients(self, gradients, loss):
        def update_stats(grad, weight):
            layer_name = weight.name
            max_abs = tf.reduce_max(tf.abs(grad))
            grad_norm = tf.norm(grad)
            # Use tf.cond to update stats in graph mode
            self.epoch_stats[layer_name]['max_abs'].assign(
                tf.maximum(self.epoch_stats[layer_name]['max_abs'], max_abs)
            )
            self.epoch_stats[layer_name]['max_norm'].assign(
                tf.maximum(self.epoch_stats[layer_name]['max_norm'], grad_norm)
            )
            self.epoch_stats[layer_name]['min_norm'].assign(
                tf.minimum(self.epoch_stats[layer_name]['min_norm'], grad_norm)
            )
            self.epoch_stats[layer_name]['mean_norm'].assign_add(grad_norm / self.n_steps)

        for grad, weight in zip(gradients, self.model.trainable_weights):
            if grad is not None : # and "_tConv4x4" not in weight.name:
                update_stats(grad, weight)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get('loss')
        if math.isnan(loss) or math.isinf(loss):
            if batch > 0:
                for layer_name in self.epoch_stats.keys(): # correct incomplete batch mean
                    self.epoch_stats[layer_name]['mean_norm'].assign(self.epoch_stats[layer_name]['mean_norm'] * self.n_steps / float(batch) )
            self.on_epoch_end(-1)

    def on_epoch_end(self, epoch, logs=None):
        max_stat = {
            'max_abs': -float('inf'),
            'max_norm': -float('inf'),
            'min_norm': float('inf'),
            'mean_norm': -float('inf')
        }
        max_l_name = {
            'max_abs': None,
            'max_norm': None,
            'min_norm': None,
            'mean_norm': None
        }
        nan = []
        inf = []
        mean_norm = 0.0
        for layer_name, stats in self.epoch_stats.items():
            for key in max_stat.keys():
                if "min" in key and stats[key].numpy() < max_stat[key] or ( ("max" in key or "mean_norm"==key) and stats[key].numpy() > max_stat[key]):
                    max_l_name[key] = layer_name
                    max_stat[key] = stats[key].numpy()
                elif "min" in key and stats[key].numpy() == max_stat[key]:
                    if not isinstance(max_l_name[key], list):
                        max_l_name[key] = [max_l_name[key]]
                    max_l_name[key].append(layer_name)
            mean_norm += stats['mean_norm'].numpy() / len(self.epoch_stats)
            if np.isnan(stats['mean_norm'].numpy()):
                nan.append(layer_name)
            elif np.isinf(stats["mean_norm"].numpy()):
                inf.append(layer_name)

        print(
            f"Gradients {f'ep={epoch + 1}' if epoch >=0 else 'ERROR'}: "
            f"Max_abs@{max_l_name['max_abs']}={max_stat['max_abs']:.6f} "
            f"Max_norm@{max_l_name['max_norm']}={max_stat['max_norm']:.6f} "
            f"Min_norm@{max_l_name['min_norm']}={max_stat['min_norm']:.8f} "
            f"Max_Mean_norm@{max_l_name['mean_norm']}={max_stat['mean_norm']:.6f} "
            f"Global mean norm ={mean_norm:.6f}",
            flush=True
        )
        if len(nan)>0:
            print(f"NaN layers: {nan}", flush=True)
        if len(inf)>0:
            print(f"inf layers: {inf}", flush=True)
        # Reset for the next epoch
        for layer_name in self.epoch_stats:
            self.epoch_stats[layer_name]['max_abs'].assign(-float('inf'))
            self.epoch_stats[layer_name]['max_norm'].assign(-float('inf'))
            self.epoch_stats[layer_name]['min_norm'].assign(float('inf'))
            self.epoch_stats[layer_name]['mean_norm'].assign(0.0)