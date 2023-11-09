import itertools
import numpy as np
from random import getrandbits, uniform, choice
import scipy.ndimage as ndi
import tensorflow as tf
def get_swim1d_function(mask_channels:list, distance:int=50, min_gap:int=3, closed_end:bool = True):
    if not isinstance(mask_channels, (list, tuple)):
        mask_channels = [mask_channels]
    assert len(mask_channels)>=1, "at least one mask channel must be provided"
    def fun(batch_by_channel):
        if distance > 1:
            channels = [c for c in batch_by_channel.keys() if not isinstance(c, str) and c>=0]
            mask_batch = batch_by_channel[mask_channels[0]]
            for b, c in itertools.product(range(mask_batch.shape[0]), range(mask_batch.shape[-1])):
                mask_img = mask_batch[b,...,c]
                # get y space between bacteria
                space_y = np.invert(np.any(mask_img, 1)).astype(int) # 1 where no label along y axis:
                space_y, n_lab = ndi.label(space_y)
                space_y = ndi.find_objects(space_y)
                space_y = [slice_obj[0] for slice_obj in space_y] # only first dim
                limit = mask_img.shape[0]
                space_y = [slice_obj for slice_obj in space_y if (slice_obj.stop - slice_obj.start)>=min_gap and slice_obj.stop>15 and (closed_end or slice_obj.start>0) and slice_obj.stop<limit] # keep only slices with length > 4 and not the space close to the open ends
                if len(space_y)>0:
                    space_y = [(slice_obj.stop + slice_obj.start)//2 for slice_obj in space_y]
                    y = choice(space_y)
                    lower = closed_end or not getrandbits(1)
                    if y==0:
                        lower = True
                    dist = uniform(min_gap, distance)
                    for chan in channels: # apply to each channel
                        translate(batch_by_channel[chan][b, ..., c:c+1], y, lower, dist, is_mask=chan in mask_channels)
    return fun

def translate(img, y, lower, dist, is_mask):
    order = 0 if is_mask else 1
    if lower:
        img[y:] = tf.keras.preprocessing.image.apply_affine_transform(img[y:], ty=-dist, row_axis=0, col_axis=1, channel_axis=2, order=order)
    else:
        img[:y] = tf.keras.preprocessing.image.apply_affine_transform(img[:y], ty=dist, row_axis=0, col_axis=1, channel_axis=2, order=order)
