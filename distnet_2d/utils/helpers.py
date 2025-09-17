import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import shutil
import edt
from dataset_iterator import MultiChannelIterator
from dataset_iterator.helpers import get_decimation_factor


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

def get_background_foreground_counts(dataset, channel_keyword:str, group_keyword:str=None, max_decimation_factor: float = None, dtype="float128"):
    iterator = MultiChannelIterator(dataset=dataset, channel_keywords=[channel_keyword], group_keyword=group_keyword, input_channels=[0], output_channels=[], batch_size=1, incomplete_last_batch_mode=0)
    f = 1
    i = 0
    counts = np.array([0, 0], dtype = dtype)
    while i < len(iterator):
        batch, = iterator[int(i)]
        if i == 0:
            f = get_decimation_factor(batch.shape, len(iterator), max_decimation_factor=max_decimation_factor)
        local_n_fore = np.sum(batch > 0)
        counts[1] += local_n_fore
        counts[0] += np.prod(batch.shape)
        i += f
    counts[0] -= counts[1] # foreground
    return counts

def count_links(previous_links, detailed:bool=False):
    """
    Count link multiplicities in forward and backward directions.

    Args:
        previous_links: Array of shape (F, N_links, 3) where:
            - F: number of frames
            - N_links: number of links per frame
            - 3: [current_label, previous_label, gap_number]
        detailed: bool return detailed counts or not

    Returns:
        dict: Counts for each multiplicity type and direction if details, else a tuple with single, multiple and null counts.
    """
    if previous_links.ndim == 2:
        previous_links = np.expand_dims(previous_links, axis=0)
    if previous_links.shape[-1] == 2:  # add gap columns
        gap_column = np.zeros(previous_links.shape[:2] + (1,), dtype=previous_links.dtype)
        previous_links = np.concatenate([previous_links, gap_column], axis=-1)

    # Initialize counters for different multiplicity types and directions
    counts = {
        'single_forward': 0,  # current_label connects to exactly one previous_label
        'single_backward': 0,  # previous_label connects to exactly one current_label
        'multiple_forward': 0,  # current_label connects to multiple previous_labels
        'multiple_backward': 0,  # previous_label connects to multiple current_labels
        'null_forward': 0,  # current_label > 0, previous_label = 0
        'null_backward': 0,  # previous_label > 0, current_label = 0
        'gap': 0  # gap links (always single by definition)
    }

    for frame in previous_links:
        current_labels = frame[:, 0]
        previous_labels = frame[:, 1]
        gap_numbers = frame[:, 2]

        # Separate gap links from regular links
        gap_mask = gap_numbers > 0
        regular_mask = gap_numbers == 0

        # Handle gap links (these are always single by definition and must have valid labels)
        if np.any(gap_mask):
            gap_current = current_labels[gap_mask]
            gap_previous = previous_labels[gap_mask]

            # Gap links must have both labels > 0 (cannot be null by definition)
            valid_gap_mask = (gap_current > 0) & (gap_previous > 0)

            # Each valid gap link is counted once (single by definition)
            counts['gap'] += np.sum(valid_gap_mask)

        # Handle regular (non-gap) links
        if np.any(regular_mask):
            regular_current = current_labels[regular_mask]
            regular_previous = previous_labels[regular_mask]

            # Identify null links with direction
            null_forward_mask = (regular_current > 0) & (regular_previous == 0)  # forward direction
            null_backward_mask = (regular_previous > 0) & (regular_current == 0)  # backward direction

            counts['null_forward'] += np.sum(null_forward_mask)
            counts['null_backward'] += np.sum(null_backward_mask)

            # Valid links (both labels > 0)
            valid_mask = (regular_current > 0) & (regular_previous > 0)

            if np.any(valid_mask):
                valid_current = regular_current[valid_mask]
                valid_previous = regular_previous[valid_mask]

                # Count how many times each current/previous label appears
                unique_current, current_counts = np.unique(valid_current, return_counts=True)
                unique_previous, previous_counts = np.unique(valid_previous, return_counts=True)

                # Create count arrays for current and previous labels
                max_current_label = np.max(valid_current) if len(valid_current) > 0 else 0
                max_previous_label = np.max(valid_previous) if len(valid_previous) > 0 else 0
                current_count_array = np.zeros(max_current_label + 1, dtype=int)
                previous_count_array = np.zeros(max_previous_label + 1, dtype=int)
                current_count_array[unique_current] = current_counts
                previous_count_array[unique_previous] = previous_counts

                # Forward direction: count from current label perspective: each unique current label contributes once
                unique_current_in_links = np.unique(valid_current)
                current_multiplicities = current_count_array[unique_current_in_links]
                counts['single_forward'] += np.sum(current_multiplicities == 1)
                counts['multiple_forward'] += np.sum(current_multiplicities > 1)

                # Backward direction: count from previous label perspective: each unique previous label contributes once
                unique_previous_in_links = np.unique(valid_previous)
                previous_multiplicities = previous_count_array[unique_previous_in_links]
                counts['single_backward'] += np.sum(previous_multiplicities == 1)
                counts['multiple_backward'] += np.sum(previous_multiplicities > 1)

    if detailed:
        return counts
    else:
        return counts['gap'] * 2 + counts['single_forward'] + counts['single_backward'], counts['multiple_forward'] + counts['multiple_backward'], counts['null_forward'] + counts['null_backward']

