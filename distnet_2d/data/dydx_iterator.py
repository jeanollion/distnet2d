from dataset_iterator import TrackingIterator
from dataset_iterator.tile_utils import extract_tile_random_zoom_function
import numpy as np
from scipy.ndimage import center_of_mass, find_objects, maximum_filter
from scipy.ndimage.measurements import mean
from skimage.transform import rescale
from math import copysign
import sys
import itertools
import edt
from random import random

class DyDxIterator(TrackingIterator):
    def __init__(self,
        dataset,
        channel_keywords:list=['/raw', '/regionLabels', '/prevRegionLabels'], # channel @1 must be label & @2 previous label
        next:bool = True,
        return_categories:bool = True,
        erase_edge_cell_size:int = 50,
        aug_remove_prob:float = 0.01,
        aug_frame_subsampling = 4, # either int: subsampling interval will be drawn uniformly in [1,aug_frame_subsampling] or callable that generate an subsampling interval (int)
        extract_tile_function = extract_tile_random_zoom_function(tile_shape=(128, 128), n_tiles=8, zoom_range=[0.6, 1.6], aspect_ratio_range=[0.75, 1.5], random_channel_jitter_shape=[50, 50] ),
        elasticdeform_parameters:dict = {},
        downscale_displacement_and_categories=1,
        input_image_data_generator=None,
        return_contours = False,
        output_float16=False,
        **kwargs):
        if len(channel_keywords)!=3:
            raise ValueError('keyword should contain 3 elements in this order: grayscale input images, object labels, object previous labels')

        self.return_categories=return_categories
        self.downscale=downscale_displacement_and_categories
        self.erase_edge_cell_size=erase_edge_cell_size
        self.aug_frame_subsampling=aug_frame_subsampling
        self.output_float16=output_float16
        self.return_contours=return_contours
        if input_image_data_generator is not None:
            kwargs["image_data_generators"] = [input_image_data_generator, None, None]
        super().__init__(dataset=dataset,
                    channel_keywords=channel_keywords,
                    input_channels=[0],
                    output_channels=[1, 2],
                    channels_prev=[True]*3,
                    channels_next=[next]*3,
                    mask_channels=[1, 2],
                    aug_remove_prob=aug_remove_prob,
                    aug_all_frames=False,
                    convert_masks_to_dtype=False,
                    extract_tile_function=extract_tile_function,
                    elasticdeform_parameters=elasticdeform_parameters,
                    **kwargs)

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False, perform_elasticdeform=True, perform_tiling=True):
        if self.aug_remove_prob>0 and random() < self.aug_remove_prob:
            self.n_frames = 0 # flag that aug_remove = true
        else:
            if self.aug_frame_subsampling>1 and self.aug_frame_subsampling is not None:
                if callable(self.aug_frame_subsampling):
                    self.n_frames = self.aug_frame_subsampling()
                else:
                    self.n_frames=np.random.randint(self.aug_frame_subsampling)+1
            else:
                self.n_frames = 1
        batch_by_channel, aug_param_array, ref_channel = super()._get_batch_by_channel(index_array, perform_augmentation, input_only, perform_elasticdeform=False, perform_tiling=False)
        if not issubclass(batch_by_channel[1].dtype.type, np.integer):
            batch_by_channel[1] = batch_by_channel[1].astype(np.int32)
        # get previous labels and store in -666 output_position BEFORE applying tiling and elastic deform
        self._get_prev_label(batch_by_channel)
        batch_by_channel[-1] = batch_by_channel[0].shape[0] # batch size is recorded here: in case of tiling it will be usefull
        del batch_by_channel[2] # remove prevRegionLabels
        if self.n_frames>1: # remove unused frames
            sel = [0, self.n_frames, -1] if self.channels_next[1] else [0, -1]
            channels = [c for c in batch_by_channel if c>=0]
            for c in channels:
                batch_by_channel[c] = batch_by_channel[c][..., sel]

        if perform_elasticdeform or perform_tiling: ## elastic deform do not support float16 type -> temporarily convert to float32
            channels = [c for c in batch_by_channel.keys() if c>=0]
            converted_from_float16=[]
            for c in channels:
                if batch_by_channel[c].dtype == np.float16:
                    batch_by_channel[c] = batch_by_channel[c].astype('float32')
                    converted_from_float16.append(c)

        if perform_elasticdeform:
            self._apply_elasticdeform(batch_by_channel)
        if perform_tiling:
            self._apply_tiling(batch_by_channel)

        if perform_elasticdeform or perform_tiling:
            for c in converted_from_float16:
                batch_by_channel[c] = batch_by_channel[c].astype('float16')

        return batch_by_channel, aug_param_array, ref_channel

    def _get_prev_label(self, batch_by_channel):
        labelIms = batch_by_channel[1]
        prevlabelIms = batch_by_channel[2]
        return_next = self.channels_next[1]
        prev_label_map = []
        n_frames = self.n_frames if self.n_frames>0 else 1
        end_points = [0, n_frames]
        if return_next:
            end_points.append(labelIms.shape[-1]-1)
        for b in range(labelIms.shape[0]):
            prev_label_map.append(_compute_prev_label_map(labelIms[b], prevlabelIms[b], end_points))
        batch_by_channel[-666] = prev_label_map

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        input = super()._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
        return_next = self.channels_next[1]
        n_frames = (input.shape[-1]-1)//2 if return_next else input.shape[-1]-1
        if n_frames>1:
            sel = [0, n_frames, -1] if return_next else [0, -1]
            return input[..., sel] # only return
        else:
            return input

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        # dx & dy are computed and returned along with edm and categories
        labelIms = batch_by_channel[1]
        return_next = self.channels_next[1]
        # remove small objects
        mask_to_erase_cur = [chan_idx for chan_idx in self.mask_channels if chan_idx!=1 and chan_idx in batch_by_channel]
        mask_to_erase_chan_cur = [1 if self.channels_prev[chan_idx] else 0 for chan_idx in mask_to_erase_cur]
        mask_to_erase_prev = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_prev[chan_idx]]
        mask_to_erase_chan_prev = [0] * len(mask_to_erase_prev)
        if return_next:
            mask_to_erase_next = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_next[chan_idx]]
            mask_to_erase_chan_next = [2 if self.channels_prev[chan_idx] else 1 for chan_idx in mask_to_erase_next]

        for i in range(labelIms.shape[0]):
            # cur timepoint
            self._erase_small_objects_at_edges(labelIms[i,...,1], i, mask_to_erase_cur, mask_to_erase_chan_cur, batch_by_channel)
            # prev timepoint
            self._erase_small_objects_at_edges(labelIms[i,...,0], i, mask_to_erase_prev, mask_to_erase_chan_prev, batch_by_channel)
            if return_next:
                self._erase_small_objects_at_edges(labelIms[i,...,-1], i, mask_to_erase_next, mask_to_erase_chan_next, batch_by_channel)

        dyIm = np.zeros(labelIms.shape[:-1]+(2 if return_next else 1,), dtype=self.dtype)
        dxIm = np.zeros(labelIms.shape[:-1]+(2 if return_next else 1,), dtype=self.dtype)
        if self.return_categories:
            categories = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
            if return_next:
                categories_next = np.zeros(labelIms.shape[:-1]+(1,), dtype=self.dtype)
        labels_map_prev = batch_by_channel[-666]
        batch_size = batch_by_channel[-1]
        if labelIms.shape[0]>batch_size:
            n_tiles = labelIms.shape[0]//batch_size
            #print("batch size: {}, n_tiles: {}".format(batch_size, n_tiles))
            get_idx = lambda x:x%batch_size # We assume here that the indexing order of tiling is tile x batch
        else:
            get_idx = lambda x:x
        for i in range(labelIms.shape[0]):
            bidx = get_idx(i)
            _compute_displacement(labelIms[i,...,:2], labels_map_prev[bidx][0], dyIm[i,...,0], dxIm[i,...,0], categories[i,...,0] if self.return_categories else None)
            if return_next:
                _compute_displacement(labelIms[i,...,1:], labels_map_prev[bidx][-1], dyIm[i,...,1], dxIm[i,...,1], categories_next[i,...,0] if self.return_categories else None)

        other_output_channels = [chan_idx for chan_idx in self.output_channels if chan_idx!=1 and chan_idx!=2]
        all_channels = [batch_by_channel[chan_idx] for chan_idx in other_output_channels]

        edm_c = 3 if return_next else 2
        edm = np.zeros(shape = labelIms.shape[:-1]+(edm_c,), dtype=np.float32)
        for b,c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
            edm[b,...,c] = edt.edt(labelIms[b,...,c], black_border=False)
        all_channels.insert(0, edm)
        if self.return_contours:
            contour = edm == 1
            all_channels.insert(1, contour)
            channel_inc = 1
        else:
            channel_inc = 0
        downscale_factor = 1./self.downscale if self.downscale>1 else 0
        scale = [1, downscale_factor, downscale_factor, 1]
        if self.downscale>>1:
            dyIm = rescale(dyIm, scale, anti_aliasing= False, order=0)
            dxIm = rescale(dxIm, scale, anti_aliasing= False, order=0)
        all_channels.insert(1+channel_inc, dyIm)
        all_channels.insert(2+channel_inc, dxIm)
        if self.return_categories:
            if self.downscale>>1:
                categories = rescale(categories, scale, anti_aliasing= False, order=0)
            all_channels.insert(3+channel_inc, categories)
            if return_next:
                if self.downscale>>1:
                    categories_next = rescale(categories_next, scale, anti_aliasing= False, order=0)
                all_channels.insert(4+channel_inc, categories_next)
        if self.output_float16:
            for i, c in enumerate(all_channels):
                if not ( self.return_contours and i==1 or i==3+channel_inc or return_next and i==4+channel_inc ): # softmax / sigmoid activation -> float32
                    all_channels[i] = c.astype('float16', copy=False)
        return all_channels

    def _erase_small_objects_at_edges(self, labelImage, batch_idx, channel_idxs, channel_idxs_chan, batch_by_channel):
        objects_to_erase = _get_small_objects_at_edges_to_erase(labelImage, self.erase_edge_cell_size)
        if len(objects_to_erase)>0:
            # erase in all mask image then in label image
            for label, slice in objects_to_erase.items():
                mask = labelImage[slice] == label
                for mask_chan_idx, c in zip(channel_idxs, channel_idxs_chan):
                    batch_by_channel[mask_chan_idx][batch_idx,...,c][slice][mask]=0
                labelImage[slice][mask] = 0

def _get_small_objects_at_edges_to_erase(labelIm, min_size):
    if min_size<=0:
        return dict()
    y_labels= set(np.unique(labelIm[[-1,0], :].ravel()))
    x_labels= set(np.unique(labelIm[:, [-1,0]].ravel()))
    edge_labels = set.union(y_labels, x_labels)
    if 0 in edge_labels:
        edge_labels.remove(0)
    if len(edge_labels)>0:
        objects = find_objects(labelIm, max_label=max(edge_labels))
        return {idx+1 : slice for idx, slice in enumerate(objects) if slice is not None and idx+1 in edge_labels and np.sum(labelIm[slice] == idx+1)<=min_size}
    else:
        return dict()

# displacement computation utils

def _get_labels_and_centers(labelIm):
    labels = np.unique(labelIm)
    if len(labels)==0:
        return [],[]
    labels = [int(round(l)) for l in labels if l!=0]
    centers = center_of_mass(labelIm, labelIm, labels)
    return dict(zip(labels, centers))

# channel dimension = frames
def _compute_prev_label_map(labelIm, prevlabelIm, end_points):
    n_chan = labelIm.shape[-1]
    for i in range(0, len(end_points)):
        if end_points[i]<0:
            end_points[i] = n_chan + end_points[i]
    min_frame=np.min(end_points)
    max_frame=np.max(end_points)
    assert min_frame<n_chan and max_frame<=n_chan, "invalid end_points"
    labels_map_prev_by_c = dict()
    labels_by_c = {c : np.unique(labelIm[...,c]) for c in range(min_frame, max_frame+1)}
    for c in range(min_frame, max_frame+1):
        prev_labels = mean(prevlabelIm[..., c], labelIm[...,c], labels_by_c[c])
        labels_map_prev_by_c[c] = {label:int(prev_label) for label, prev_label in zip(labels_by_c[c], prev_labels)}
    labels_map_prev = []
    for i in range(len(end_points)-1):
        start = end_points[i]
        stop = end_points[i+1]
        assert stop>start, "invalid endpoint [{}, {}]".format(start, stop)
        if start == stop-1: # successive frames: prev labels are simply those of prevlabelIm
            labels_map_prev.append(labels_map_prev_by_c[stop])
        else: # frame subsampling -> iterate through lineage to get the previous label @ last frame
            labels_map_prev_cur = labels_map_prev_by_c[stop]
            #print("endpoint lmp @ {} = {}".format(stop, labels_map_prev_))
            for c in range(stop-1, start, -1):
                labels_map_prev_temp = labels_map_prev_by_c[c]
                get_prev = lambda label : labels_map_prev_temp[label] if label in labels_map_prev_temp else 0
                labels_map_prev_cur = {label:get_prev(prev) for label,prev in labels_map_prev_cur.items()}
                #print("lmp @ {} = {}".format(c, labels_map_prev_cur))
            labels_map_prev.append(labels_map_prev_cur)
    return labels_map_prev

def _compute_displacement(labelIm, labels_map_prev, dyIm, dxIm, categories=None):
    labels_map_centers = [_get_labels_and_centers(labelIm[...,c]) for c in range(labelIm.shape[-1])]
    if len(labels_map_centers[-1])==0:
        return
    curLabelIm = labelIm[...,-1]
    labels_prev = labels_map_centers[0].keys()
    for label, center in labels_map_centers[-1].items():
        label_prev = labels_map_prev[label]
        if label_prev in labels_prev:
            dy = center[0] - labels_map_centers[0][label_prev][0] # axis 0 is y
            dx = center[1] - labels_map_centers[0][label_prev][1] # axis 1 is x
            if categories is None and abs(dy)<1:
                dy = copysign(1, dy) # min value == 1 / same sign as dy
            if categories is None and abs(dx)<1:
                dx = copysign(1, dx) # min value == 1 / same sign as dx
            dyIm[curLabelIm == label] = dy
            dxIm[curLabelIm == label] = dx

    if categories is not None:
        labels = labels_map_centers[-1].keys()
        labels_of_prev = [labels_map_prev[l] for l in labels]
        labels_of_prev_counts = dict(zip(*np.unique(labels_of_prev, return_counts=True)))
        for label, label_prev in zip(labels, labels_of_prev):
            if label_prev>0 and label_prev not in labels_prev: # no previous
                value=3
            elif labels_of_prev_counts.get(label_prev, 0)>1: # division
                value=2
            else: # previous has single next
                value=1
            categories[curLabelIm == label] = value
