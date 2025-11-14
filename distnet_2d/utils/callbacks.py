import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

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
        self.initial_weights = tf.identity(self._get().numpy())
        self.final_weights = tf.ones_like(self.initial_weights)

    def _get(self):
        var = getattr(self.model, self.attribute_name)
        if isinstance(var, HideVariableWrapper):
            return var.value
        else:
            return var

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        alpha = tf.cast( tf.math.pow( float(epoch) / self.n_epochs, self.power_law ), self.initial_weights.dtype)
        cur_weights = self.initial_weights * (1 - alpha) + self.final_weights * alpha
        self._get().assign( cur_weights )
        if self.verbose and epoch % 100 == 0:
            print(f"Epoch: {epoch} weights for {self.attribute_name}: {self._get().numpy()}", flush=True)
