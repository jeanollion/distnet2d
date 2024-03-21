from math import cos, pi
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
import csv
import os

class StopOnLR(Callback):
    def __init__( self, min_lr, **kwargs, ):
        super().__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs={}):
        lr = backend.get_value(self.model.optimizer.lr)
        if(lr <= self.min_lr):
            self.model.stop_training = True

class EpsilonCosineDecayCallback(Callback):
    """Reduce optimizer epsilon parameter.
    Args:
      factor: factor by which epsilon will be reduced.
        `new_epsilon = epsilon * factor`.
      period: number of epochs after which epsilon will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      min_epsilon: lower bound on the learning rate.
    """

    def __init__(self, decay_steps:int, start_epsilon=1, min_epsilon=1e-2, start_step:int=0, verbose=1):
        super(EpsilonCosineDecayCallback, self).__init__()
        self.step_counter = start_step
        self.start_epsilon = start_epsilon
        self.decay_steps=decay_steps
        self.alpha=min_epsilon / start_epsilon
        self.verbose = verbose

    def on_train_batch_begin(self, batch, logs=None):
        epsilon = self._decayed_epsilon(self.step_counter)
        self.model.optimizer.epsilon = epsilon

    def on_train_batch_end(self, batch, logs=None):
        self.step_counter +=1
        if logs is not None:
            logs['epsilon'] = self.model.optimizer.epsilon

    def _decayed_epsilon(self, step):
        step = min(step, self.decay_steps)
        cosine_decay = 0.5 * (1 + cos(pi * step / self.decay_steps))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        return self.start_epsilon * decayed

class LogsCallback(Callback):
    def __init__(self, filepath, start_epoch=0):
        super().__init__()
        self.filepath = filepath
        self.start_epoch=start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            epoch += self.start_epoch
            losses = list(logs.values())
            losses.insert(0, epoch)
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            with open(filepath, "a") as file_object:
                writer = csv.writer(file_object, delimiter=';',  quotechar='"', quoting=csv.QUOTE_MINIMAL)
                if os.path.getsize(filepath) == 0: # write header only if file is empty
                    headers = list(logs.keys())
                    headers.insert(0, "epoch")
                    writer.writerow(headers)
                writer.writerow(losses)
