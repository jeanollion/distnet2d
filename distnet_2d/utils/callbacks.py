from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend

class StopOnLR(Callback):
    def __init__( self, min_lr, **kwargs, ):
        super().__init__()
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs={}):
        lr = backend.get_value(self.model.optimizer.lr)
        if(lr <= self.min_lr):
            self.model.stop_training = True
