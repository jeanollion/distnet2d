import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import shutil

def convert_probabilities_to_logits(y_pred): # y_pred should be a tensor: tf.convert_to_tensor(y_pred, np.float32)
      y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
      return K.log(y_pred / (1 - y_pred))

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
    if isinstance(element, list):
        l.extend(element)
    else:
        l.append(element)

def get_earse_small_values_function(thld):
    def earse_small_values(im):
        im[im<thld]=0
        return im
    return earse_small_values

def step_decay_schedule(initial_lr=1e-3, minimal_lr=1e-5, decay_factor=0.50, step_size=50):
    if minimal_lr>initial_lr:
        raise ValueError("Minimal LR should be inferior to initial LR")
    def schedule(epoch):
        lr = max(initial_lr * (decay_factor ** np.floor(epoch/step_size)), minimal_lr)
        return lr
    return LearningRateScheduler(schedule, verbose=1)


def evaluate_model(iterator, model, metrics, metric_names, xp_idx_in_path=2, position_idx_in_path=3, progress_callback=None):
    try:
        import pandas as pd
    except ImportError as error:
        print("Pandas not installed")
        return

    arr, paths, labels, indices = iterator.evaluate(model, metrics, progress_callback=progress_callback)
    df = pd.DataFrame(arr)
    if len(metric_names)+2 != df.shape[1]:
        raise ValueError("Invalid loss / accuracy name: expected: {} names, got: {} names".format(df.shape[1]-2, len(metric_names)))
    df.columns=["Idx", "dsIdx"]+metric_names
    df["Indices"] = pd.Series(indices)
    dsInfo = np.asarray([p.split('/') for p in paths])
    df["XP"] = pd.Series(dsInfo[:,xp_idx_in_path])
    df["Position"] = pd.Series(dsInfo[:,position_idx_in_path])
    return df

def displayProgressBar(max): # this progress bar is compatible with google colab
    from IPython.display import HTML, display
    def progress(value=0, max=max):
        return HTML("""
            <progress
                value='{value}'
                max='{max}',
                style='width: 100%'
            >
                {value}
            </progress>
        """.format(value=value, max=max))
    out = display(progress(), display_id=True)
    currentProgress=[0]
    def callback():
        currentProgress[0]+=1
        out.update(progress(currentProgress[0]))
    return callback

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

def get_nd_gaussian_kernel(radius=1, sigma=0, ndim=2):
    size = 2 * radius + 1
    if ndim == 1:
        coords = [np.mgrid[-radius:radius:complex(0, size)]]
    elif ndim==2:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    elif ndim==3:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    elif ndim==4:
        coords = np.mgrid[-radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size), -radius:radius:complex(0, size)]
    else:
        raise ValueError("Up to 4D supported")

    # Need an (N, ndim) array of coords pairs.
    stacked = np.column_stack([c.flat for c in coords])
    mu = np.array([0.0]*ndim)
    s = np.array([sigma if sigma>0 else radius]*ndim)
    covariance = np.diag(s**2)
    z = multivariate_normal.pdf(stacked, mean=mu, cov=covariance)
    z = z.reshape(coords[0].shape) # Reshape back to a (N, N) grid.
    return z/z.sum()
