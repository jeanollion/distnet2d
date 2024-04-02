from math import cos, pi
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras import backend
import csv
import os
import time


class SafeModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 n_retry:int = 10,
                 sleep_time:int = 5,
                 **kwargs):
        super().__init__(filepath, **kwargs)
        self.n_retry = n_retry
        self.sleep_time = sleep_time
        self._alternative_path = False

    def _save_model(self, epoch, batch, logs):
        for i in range(self.n_retry):
            try:
                if i == self.n_retry - 1:
                    self._alternative_path = True
                super()._save_model(epoch, batch, logs)
                if i>1:
                    print(f"model saved after retrying: {i+1}/{self.n_retry}", flush=True)
            except BlockingIOError as error:
                print(f"Error saving weights: {error}. Retry: {i+1}/{self.n_retry}", flush=True)
                time.sleep(self.sleep_time)  # wait for X seconds before trying to save again
                os.remove(self._get_file_path(epoch, batch, logs))
                self._alternative_path = False
            else:
                self._alternative_path = False
                return

        # 21:45:07.667:   File "h5py/h5f.pyx", line 126, in h5py.h5f.create
        # 21:45:07.667: BlockingIOError: [Errno 11] Unable to create file (unable to lock file, errno = 11, error message = 'Resource temporarily unavailable')

    def _get_file_path(self, epoch, batch, logs):
        path = super()._get_file_path(epoch, batch, logs)
        if self._alternative_path:
            split = os.path.splitext(path)
            path = split[0]+"_tmp"+split[1]
        return path

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
