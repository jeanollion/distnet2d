import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback


class ClassWeightScheduler(Callback):
    def __init__(self, attribute_name:str, n_epochs:int, power_law:float=1., verbose:bool=True):
        super().__init__()
        self.n_epochs = n_epochs
        self.power_law = power_law
        self.attribute_name = attribute_name
        self.initial_weights = None
        self.final_weights = None
        self.verbose=verbose

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)
        self.initial_weights = self._get().numpy()
        self.final_weights = np.ones_like(self.initial_weights)

    def _get(self):
        return getattr(self.model, self.attribute_name)

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)
        alpha = ( epoch / self.n_epochs ) ** self.power_law
        cur_weights = self.initial_weights * (1 - alpha) + self.final_weights * alpha # linear interpolation
        if self.verbose and epoch%100==0:
            print(f"Epoch: {epoch} weights for {self.attribute_name}: {cur_weights}", flush=True)
        self._get().assign( cur_weights )
