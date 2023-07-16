import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import shutil
import edt
from dataset_iterator import MultiChannelIterator

def ensure_multiplicity(n, object):
    if object is None:
        return [None] * n
    if not isinstance(object, (list, tuple)):
        object = [object]
    if len(object)>1 and len(object)!=n:
        raise ValueError("length should be either 1 either {}".format(n))
    if n>1 and len(object)==1:
        object = object*n
    elif n==0:
        return []
    return object

def flatten_list(l):
    flat_list = []
    for item in l:
        append_to_list(flat_list, item)
    return flat_list

def append_to_list(l, element):
    if isinstance(element, tuple):
        element = list(element)
    if isinstance(element, list):
        l.extend(element)
    else:
        l.append(element)

def step_decay_schedule(initial_lr=1e-3, minimal_lr=1e-5, decay_factor=0.50, step_size=50):
    if minimal_lr>initial_lr:
        raise ValueError("Minimal LR should be inferior to initial LR")
    def schedule(epoch):
        lr = max(initial_lr * (decay_factor ** np.floor(epoch/step_size)), minimal_lr)
        return lr
    return LearningRateScheduler(schedule, verbose=1)

def predict_average_flip_rotate(model, batch, allow_permute_axes = True, training=False):
    list_flips=[0,1,2] if allow_permute_axes else [0, 1]
    batch_list = _append_flip_and_rotate_list(batch, list_flips)
    if training is None:
        predicted_list = [model(b) for b in batch_list]
    else:
        predicted_list = [model(b, training=training) for b in batch_list]
    # transform back
    if isinstance(predicted_list[0], (tuple, list)):
        predicted_list = _transpose(predicted_list)
        return tuple([_reverse_and_mean(l, list_flips) for l in predicted_list])
    else:
        return _reverse_and_mean(predicted_list, list_flips)

def _append_flip_and_rotate_list(batch, list_transfo):
    if isinstance(batch, (tuple, list)):
        batch_list = []
        for i in range(len(batch)):
            batch_list.append(_append_flip_and_rotate(batch, list_transfo))
        return _transpose(batch_list)
    else:
        return _append_flip_and_rotate(batch, list_transfo)

def _append_flip_and_rotate(batch, list_transfo):
    trans = [batch] + [AUG_FUN_2D[transfo_idx](batch) for transfo_idx in list_transfo]
    return trans

def _reverse_and_mean(image_list, list_transfo):
    n_flips = len(list_transfo)
    for idx, transfo_idx in enumerate(list_transfo):
        image_list[idx+1] = AUG_FUN_2D[transfo_idx](image_list[idx+1])
    return np.mean(image_list, axis=0)

def _transpose(list_of_list):
    size1=len(list_of_list)
    size2=len(list_of_list[0])
    return [ [ list_of_list[i][j] for i in range(size1)] for j in range(size2) ]

AUG_FUN_2D = [
    lambda img : np.flip(img, axis=1),
    lambda img : np.flip(img, axis=2),
    lambda img : np.transpose(img, axes=(0, 2, 1, 3))
]
