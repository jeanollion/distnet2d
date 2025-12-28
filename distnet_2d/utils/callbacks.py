import math
from math import cos, pi
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from collections import defaultdict

from tensorflow.keras.optimizers.schedules import CosineDecay, LearningRateSchedule

from distnet_2d.model.layers import HideVariableWrapper, ScheduledDropout, ScheduledGradientWeight, \
    ResidualGradientLimiter


class CosineDecayResume(LearningRateSchedule):
    def __init__(
            self,
            initial_learning_rate:float,
            decay_steps:int,
            alpha:float=0.0,
            start_step:int=0,
            warmup_learning_rate_factor:float=None,
            warmup_steps:int = 0,
            name="CosineDecayResume"):
        super().__init__()
        self.start_step = int(start_step)
        self.initial_learning_rate = float(initial_learning_rate)
        self.warmup_learning_rate_factor = warmup_learning_rate_factor
        self.warmup_steps = int(warmup_steps)
        self.cosine_decay = CosineDecay(initial_learning_rate, decay_steps-self.warmup_steps, alpha, name=name)

    def __call__(self, step):
        if self.warmup_steps>0 and self.warmup_learning_rate_factor is not None:
            return tf.cond(step<self.warmup_steps, lambda: self.warmup(step), lambda: self.cosine_decay(step + self.start_step - self.warmup_steps))
        else:
            return self.cosine_decay(step + self.start_step - self.warmup_steps)

    def warmup(self, step):
        target_learning_rate = self.cosine_decay(tf.maximum(tf.cast(0, tf.int32), tf.cast(step + self.start_step - self.warmup_steps, tf.int32)))
        dtype = target_learning_rate.dtype
        warmup_lr = tf.cast(self.warmup_learning_rate_factor, dtype=dtype) * target_learning_rate
        step = tf.cast(step, dtype=dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype=dtype)
        return warmup_lr + (target_learning_rate - warmup_lr) * step / warmup_steps


class EpsilonCosineDecayCallback(Callback):
    """Reduce optimizer epsilon parameter.
    Args:
      factor: factor by which epsilon will be reduced.
        `new_epsilon = epsilon * factor`.
      period: number of epochs after which epsilon will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      min_epsilon: lower bound on the learning rate.
    """

    def __init__(self, decay_steps:int, start_epsilon=1e-6, min_epsilon=1e-7, start_step:int=0, verbose=1):
        super(EpsilonCosineDecayCallback, self).__init__()
        self.step_counter = start_step
        self.start_epsilon = start_epsilon
        self.decay_steps=decay_steps
        self.alpha=min_epsilon / start_epsilon
        self.verbose = verbose

    def on_train_batch_begin(self, batch, logs=None):
        epsilon = self._decayed_epsilon(self.step_counter)
        self._set_epsilon(epsilon)

    def on_train_batch_end(self, batch, logs=None):
        self.step_counter +=1
        if logs is not None:
            logs['epsilon'] = self._get_epsilon()

    def _decayed_epsilon(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.start_epsilon * decayed

    def _get_epsilon(self):
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'inner_optimizer'):
            optimizer = optimizer.inner_optimizer
        return optimizer.epsilon

    def _set_epsilon(self, epsilon):
        optimizer = self.model.optimizer
        if hasattr(optimizer, 'inner_optimizer'):
            optimizer = optimizer.inner_optimizer
        optimizer.epsilon = epsilon

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


class ScheduledDropoutCallback(tf.keras.callbacks.Callback):
    """
    Automatically discovers ScheduledDropout layers and updates their progress
    based on training epochs. Each layer handles its own schedule via max_progress.
    """

    def __init__(self, n_epochs, verbose=1):
        """
        Args:
            n_epochs: Total number of training epochs
            verbose: 0, 1, or 2 for logging level
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.dropout_layers = []

    @staticmethod
    def _find_dropout_layers(layer):
        """
        Recursively find all ScheduledDropout layers in a model.

        Args:
            layer: A Keras layer or model

        Returns:
            List of ScheduledDropout layers
        """
        if isinstance(layer, ScheduledDropout):
            return [layer]

        # Handle nested models/layers
        layers = []
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                layers.extend(ScheduledDropoutCallback._find_dropout_layers(sublayer))
        return layers

    def on_train_begin(self, logs=None):
        """Scan model and register all ScheduledDropout layers"""
        self.dropout_layers = self._find_dropout_layers(self.model)

        if self.verbose > 0:
            print(f"\n{'=' * 70}")
            print(f"CurriculumDropoutScheduler initialized")
            print(f"{'=' * 70}")
            print(f"Found {len(self.dropout_layers)} ScheduledDropout layer(s)")
            print(f"Total epochs: {self.n_epochs}")
            print(f"\nLayer configurations:")

            for i, layer in enumerate(self.dropout_layers):
                completion_epoch = int(layer.max_progress * self.n_epochs)
                print(f"  {layer.name}:")
                print(f"    min_rate={layer.min_rate:.3f}, max_rate={layer.max_rate:.3f}")
                print(f"    max_progress={layer.max_progress:.3f} → completes at epoch {completion_epoch}")

            print(f"{'=' * 70}\n")

    def on_epoch_begin(self, epoch, logs=None):
        """Update global progress at the beginning of each epoch"""
        if not self.dropout_layers:
            return

        # Global training progress [0, 1]
        global_progress = epoch / self.n_epochs if self.n_epochs > 0 else 1.

        # Update all layers with the same global progress
        # Each layer will compute its own rate based on its max_progress
        for layer in self.dropout_layers:
            layer.set_progress(global_progress)
            if self.verbose > 0 and epoch % 100  == 0:
                print(f"epoch: {epoch} layer: {layer.name} dropout rate: {layer.get_current_rate()}")


class ScheduledGradientCallback(tf.keras.callbacks.Callback):
    """
    Automatically discovers ScheduledGradientWeight layers and updates their progress
    based on training epochs. Each layer handles its own schedule via max_progress.
    """

    def __init__(self, n_epochs, verbose=1):
        """
        Args:
            n_epochs: Total number of training epochs
            verbose: 0, 1, or 2 for logging level
        """
        super().__init__()
        self.n_epochs = n_epochs
        self.verbose = verbose

        self.gradient_layers = []
        self.gradient_limiter_layers = []

    @staticmethod
    def _find_gradient_layers(layer):
        """
        Recursively find all ScheduledGradientWeight layers in a model.

        Args:
            layer: A Keras layer or model

        Returns:
            List of ScheduledGradientWeight layers
        """
        if isinstance(layer, ScheduledGradientWeight):
            return [layer]

        # Handle nested models/layers
        layers = []
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                layers.extend(ScheduledGradientCallback._find_gradient_layers(sublayer))
        return layers

    @staticmethod
    def _find_gradient_limiter_layers(layer):
        """
        Recursively find all ScheduledGradientWeight layers in a model.

        Args:
            layer: A Keras layer or model

        Returns:
            List of ScheduledGradientWeight layers
        """
        if isinstance(layer, ResidualGradientLimiter):
            return [layer]

        # Handle nested models/layers
        layers = []
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                layers.extend(ScheduledGradientCallback._find_gradient_limiter_layers(sublayer))
        return layers

    def on_train_begin(self, logs=None):
        """Scan model and register all ScheduledGradientWeight layers"""
        self.gradient_layers = self._find_gradient_layers(self.model)

        if self.verbose > 0:
            print(f"\n{'=' * 70}")
            print(f"ScheduledGradientCallback initialized")
            print(f"{'=' * 70}")
            print(f"Found {len(self.gradient_layers)} ScheduledGradientWeight layer(s)")
            print(f"Total epochs: {self.n_epochs}")
            print(f"\nLayer configurations:")

            for i, layer in enumerate(self.gradient_layers):
                completion_epoch = int(layer.max_progress * self.n_epochs)
                print(f"  {layer.name}:")
                print(f"    min_weight={layer.min_weight:.3f}, max_weight={layer.max_weight:.3f}")
                print(f"    max_progress={layer.max_progress:.3f} → completes at epoch {completion_epoch}")

            print(f"{'=' * 70}\n")

        self.gradient_limiter_layers = self._find_gradient_limiter_layers(self.model)

    def on_epoch_begin(self, epoch, logs=None):
        """Update global progress at the beginning of each epoch"""
        if not self.gradient_layers:
            return

        # Global training progress [0, 1]
        global_progress = epoch / self.n_epochs if self.n_epochs > 0 else 1.

        # Update all layers with the same global progress
        # Each layer will compute its own weight based on its max_progress
        for layer in self.gradient_layers:
            layer.set_progress(global_progress)
            if self.verbose > 0 and epoch % 100 == 0:
                print(f"epoch: {epoch} layer: {layer.name} gradient weight: {layer.get_current_weight()}")

        if self.verbose > 0 and epoch % 100 == 1:
            for layer in self.gradient_limiter_layers:
                print(f"epoch: {epoch} limiter: {layer.name} EMA : {layer.get_ema_values()}")
