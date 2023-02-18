from dataset_iterator import TrackingIterator
from dataset_iterator.tile_utils import extract_tile_random_zoom_function
import numpy as np
import numpy.ma as ma
from scipy.ndimage import center_of_mass, find_objects, maximum_filter, map_coordinates
from scipy.ndimage.measurements import mean
from skimage.transform import rescale
from skimage.feature import peak_local_max
from math import copysign, isnan
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
        aug_frame_subsampling = 4, # either int: subsampling interval will be drawn uniformly in [frame_window,aug_frame_subsampling] or callable that generate an subsampling interval (int)
        frame_window=1,
        extract_tile_function = extract_tile_random_zoom_function(tile_shape=(128, 128), n_tiles=8, zoom_range=[0.6, 1.6], aspect_ratio_range=[0.75, 1.5], random_channel_jitter_shape=[50, 50] ),
        elasticdeform_parameters:dict = {},
        downscale_displacement_and_categories=1,
        return_contours = False,
        contour_sigma = 0.5,
        return_center = False,
        center_mode = "GEOMETRICAL", # GEOMETRICAL, "EDM_MAX", "EDM_MEAN", "SKELETON"
        return_label_rank = False,
        output_float16=False,
        **kwargs):
        assert len(channel_keywords)==3, 'keyword should contain 3 elements in this order: grayscale input images, object labels, object previous labels'

        self.return_categories=return_categories
        self.downscale=downscale_displacement_and_categories
        self.erase_edge_cell_size=erase_edge_cell_size
        self.aug_frame_subsampling=aug_frame_subsampling
        self.output_float16=output_float16
        self.return_contours=return_contours
        self.contour_sigma=contour_sigma
        self.return_center=return_center
        self.center_mode=center_mode
        self.return_label_rank=return_label_rank
        assert frame_window>=1, "frame_window must be >=1"
        self.frame_window = frame_window
        self.n_label_max = kwargs.pop("n_label_max", 2000)
        super().__init__(dataset=dataset,
                    channel_keywords=channel_keywords,
                    input_channels=[0],
                    output_channels=[1, 2],
                    channels_prev=[True]*3,
                    channels_next=[next]*3,
                    mask_channels=[1, 2],
                    n_frames = self.frame_window,
                    aug_remove_prob=aug_remove_prob,
                    aug_all_frames=False,
                    convert_masks_to_dtype=False,
                    extract_tile_function=extract_tile_function,
                    elasticdeform_parameters=elasticdeform_parameters,
                    **kwargs)

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False, perform_elasticdeform=True, perform_tiling=True, **kwargs):
        if self.aug_remove_prob>0 and random() < self.aug_remove_prob:
            n_frames = 0 # flag that aug_remove = true
        else:
            if self.aug_frame_subsampling is not None :
                if callable(self.aug_frame_subsampling):
                    n_frames = self.aug_frame_subsampling()
                elif self.aug_frame_subsampling>1:
                    n_frames = np.random.randint(self.aug_frame_subsampling)+self.frame_window
                else:
                    n_frames = self.frame_window
            else:
                n_frames = self.frame_window
        kwargs.update({"n_frames":n_frames})
        batch_by_channel, aug_param_array, ref_channel = super()._get_batch_by_channel(index_array, perform_augmentation, input_only, perform_elasticdeform=False, perform_tiling=False, **kwargs)
        if not issubclass(batch_by_channel[1].dtype.type, np.integer): # label
            batch_by_channel[1] = batch_by_channel[1].astype(np.int32)
        # correction for oob edm_c = np.where(edm>0, np.exp(-np.square(edm-1)/2), 0.)@ previous labels
        for b in range(batch_by_channel[2].shape[0]):
            for i in range(n_frames):
                inc = n_frames - i
                prev_inc = aug_param_array[b][ref_channel].get(f"prev_inc_{inc}", inc)
                if prev_inc!=inc:
                    #print(f"oob prev: n_frames={n_frames}, idx:{i} inc={inc} actual inc:{prev_inc} will replace at {i+1}")
                    batch_by_channel[2][b][...,i+1] = batch_by_channel[1][b][...,i]
                if self.channels_next[1]:
                    next_inc = aug_param_array[b][ref_channel].get(f"next_inc_{inc}", inc)
                    if next_inc!=inc:
                        #print(f"oob next: n_frames={n_frames}, idx:{i} inc={inc} actual inc:{next_inc} will replace prev labels at {n_frames+inc}")
                        batch_by_channel[2][b][...,n_frames+inc] = batch_by_channel[1][b][...,n_frames+inc-1]
        # get previous labels and store in -666 output_position BEFORE applying tiling and elastic deform
        self._get_prev_label(batch_by_channel, n_frames)
        batch_by_channel[-1] = batch_by_channel[0].shape[0] # batch size is recorded here: in case of tiling it will be usefull
        del batch_by_channel[2] # remove prevRegionLabels
        if n_frames>1: # remove unused frames
            sel = self._get_end_points(n_frames, False)
            channels = [c for c in batch_by_channel if c>=0]
            #print(f"remove unused frames: nframes: {n_frames*2+1} sel: {sel} channels: {channels}")
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

    def _get_frames_to_augment(self, img, chan_idx, aug_params):
        if self.aug_all_frames:
            return list(range(img.shape[-1]))
        n_frames = (img.shape[-1]-1)//2 if self.channels_prev[chan_idx] and self.channels_next[chan_idx] else img.shape[-1]-1
        return self._get_end_points(n_frames, False)

    def _get_end_points(self, n_frames, pairs):
        return_next = self.channels_next[1]
        end_points = [int(n_frames * i/self.frame_window + 0.5) for i in range(0, self.frame_window+1)]
        if return_next:
            end_points = end_points + [n_frames + int(n_frames * i/self.frame_window + 0.5) for i in range(1, self.frame_window+1)]
        if pairs:
            end_point_pairs = [[end_points[i], n_frames] for i in range(0, self.frame_window)]
            if return_next:
                end_point_pairs = end_point_pairs + [[n_frames, end_points[i+self.frame_window+1]] for i in range(0, self.frame_window)]
            return end_point_pairs
        else:
            return end_points

    def _get_prev_label(self, batch_by_channel, n_frames):
        labelIms = batch_by_channel[1]
        prevlabelIms = batch_by_channel[2]
        return_next = self.channels_next[1]
        prev_label_map = []
        if n_frames <=0:
            n_frames = self.frame_window
        assert labelIms.shape[-1]==prevlabelIms.shape[-1] and labelIms.shape[-1]==1+n_frames*(2 if return_next else 1), f"invalid channel number: labels: {labelIms.shape[-1]} prev labels: {prevlabelIms.shape[-1]} n_frames: {n_frames}"
        end_point_pairs = self._get_end_points(n_frames, True)
        #print(f"n_frames: {n_frames}, nchan: {labelIms.shape[-1]}, frame_window: {self.frame_window}, return_next: {return_next}, end_points: {self._get_end_points(n_frames, False)}, end_point_pairs: {end_point_pairs}")
        for b in range(labelIms.shape[0]):
            prev_label_map.append(_compute_prev_label_map(labelIms[b], prevlabelIms[b], end_point_pairs))
        batch_by_channel[-666] = prev_label_map

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        # dx & dy are computed and returned along with edm and categories
        labelIms = batch_by_channel[1]
        return_next = self.channels_next[1]

        # remove small object
        mask_to_erase_cur = [chan_idx for chan_idx in self.mask_channels if chan_idx!=1 and chan_idx in batch_by_channel]
        mask_to_erase_chan_cur = [self.frame_window if self.channels_prev[chan_idx] else 0 for chan_idx in mask_to_erase_cur]
        mask_to_erase_prev = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_prev[chan_idx]]
        mask_to_erase_chan_prev = [0] * len(mask_to_erase_prev)
        if return_next:
            mask_to_erase_next = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_next[chan_idx]]
            mask_to_erase_chan_next = [self.frame_window+1 if self.channels_prev[chan_idx] else 1 for chan_idx in mask_to_erase_next]
        for i in range(labelIms.shape[0]):
            # cur timepoint
            self._erase_small_objects_at_edges(labelIms[i,...,self.frame_window], i, mask_to_erase_cur, mask_to_erase_chan_cur, batch_by_channel)
            # prev timepoints
            for j in range(0, self.frame_window):
                self._erase_small_objects_at_edges(labelIms[i,...,j], i, mask_to_erase_prev, [m+j for m in mask_to_erase_chan_prev], batch_by_channel)
            if return_next:
                for j in range(0, self.frame_window):
                    self._erase_small_objects_at_edges(labelIms[i,...,self.frame_window+1+j], i, mask_to_erase_next, [m+j for m in mask_to_erase_chan_next], batch_by_channel)

        edm = np.zeros(shape = labelIms.shape, dtype=np.float32)
        for b,c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
            edm[b,...,c] = edt.edt(labelIms[b,...,c], black_border=False)

        dyIm = np.zeros(labelIms.shape[:-1]+(2 * self.frame_window if return_next else self.frame_window,), dtype=self.dtype)
        dxIm = np.zeros(labelIms.shape[:-1]+(2 * self.frame_window if return_next else self.frame_window,), dtype=self.dtype)
        centerIm = np.zeros(labelIms.shape, dtype=self.dtype) if self.return_center else None
        if self.return_label_rank:
            labelIm = np.zeros(labelIms.shape, dtype=np.int32)
            # noNextArr = np.zeros(labelIms.shape[:1]+(2 * self.frame_window if return_next else self.frame_window, self.n_label_max), dtype=np.bool)
            prevLabelArr = np.zeros(labelIms.shape[:1]+(2 * self.frame_window if return_next else self.frame_window, self.n_label_max), dtype=np.int32)
        if self.return_categories:
            categoryIm = np.zeros(labelIms.shape[:-1]+(2 * self.frame_window if return_next else self.frame_window,), dtype=self.dtype)
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
            for c in range(0, self.frame_window):
                sel = [c, self.frame_window]
                _compute_displacement(labelIms[i][...,sel], labels_map_prev[bidx][c], dyIm[i,...,c], dxIm[i,...,c], edm[i][...,sel], center_mode=self.center_mode, centerIm=centerIm[i,...,self.frame_window] if self.return_center and c==0 else None, centerImPrev=centerIm[i,...,c] if self.return_center else None, categoryIm=categoryIm[i,...,c] if self.return_categories else None, rankIm=labelIm[i,...,self.frame_window] if self.return_label_rank and c==0 else None, rankImPrev=labelIm[i,...,c] if self.return_label_rank else None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank else None)
            if return_next:
                for c in range(self.frame_window, 2*self.frame_window):
                    sel = [self.frame_window, c+1]
                    _compute_displacement(labelIms[i][...,sel], labels_map_prev[bidx][c], dyIm[i,...,c], dxIm[i,...,c], edm[i][...,sel], center_mode=self.center_mode, centerIm=centerIm[i,...,c+1] if self.return_center else None, centerImPrev=None, categoryIm=categoryIm[i,...,c] if self.return_categories else None, rankIm=labelIm[i,...,c+1] if self.return_label_rank else None, rankImPrev=None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank else None)

        other_output_channels = [chan_idx for chan_idx in self.output_channels if chan_idx!=1 and chan_idx!=2]
        all_channels = [batch_by_channel[chan_idx] for chan_idx in other_output_channels]
        channel_inc = 0
        all_channels.insert(channel_inc, edm)
        if self.return_contours:
            channel_inc += 1
            #contour = edm == 1
            mul = - 1. / (self.contour_sigma * self.contour_sigma)
            contour = np.where(edm>0, np.exp(np.square(edm-1) * mul), 0.)
            all_channels.insert(channel_inc, contour)
        if self.return_center:
            channel_inc+=1
            all_channels.insert(channel_inc, centerIm)
        downscale_factor = 1./self.downscale if self.downscale>1 else 0
        scale = [1, downscale_factor, downscale_factor, 1]
        if self.downscale>1:
            dyIm = rescale(dyIm, scale, anti_aliasing= False, order=0)
            dxIm = rescale(dxIm, scale, anti_aliasing= False, order=0)
        all_channels.insert(1+channel_inc, dyIm)
        all_channels.insert(2+channel_inc, dxIm)
        channel_inc+=2
        if self.return_categories:
            channel_inc+=1
            if self.downscale>1:
                categoryIm = rescale(categoryIm, scale, anti_aliasing= False, order=0)
            all_channels.insert(channel_inc, categoryIm)
        if self.return_label_rank:
            channel_inc+=1
            all_channels.insert(channel_inc, labelIm)
            all_channels.insert(channel_inc, prevLabelArr)
        if self.output_float16:
            for i, c in enumerate(all_channels):
                if not ( self.return_contours and i==1 or self.return_categories and i==3+channel_inc or self.return_label_rank and (i==channel_inc or i==channel_inc+1) ): # softmax / sigmoid activation -> float32
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

def _get_labels_and_centers(labelIm, edm, center_mode = "GEOMETRICAL"):
    labels = np.unique(labelIm)
    labels = [int(round(l)) for l in labels if l!=0]
    if len(labels)==0:
        return dict()
    if center_mode == "GEOMETRICAL":
        centers = center_of_mass(labelIm, labelIm, labels)
    elif center_mode == "EDM_MAX":
        assert edm is not None and edm.shape == labelIm.shape
        centers = []
        for label in labels:
            edm_label = ma.array(edm, mask = labelIm != label)
            center = ma.argmax(edm_label, fill_value=0)
            center = np.unravel_index(center, edm_label.shape)
            centers.append(center)
    elif center_mode == "EDM_MEAN":
        assert edm is not None and edm.shape == labelIm.shape
        centers = center_of_mass(edm, labelIm, labels)
    elif center_mode == "SKELETON":
        assert edm is not None and edm.shape == labelIm.shape
        mass_centers = np.array(center_of_mass(labelIm, labelIm, labels))[np.newaxis] # 1, N_ob, 2
        lm_coords = peak_local_max(edm, labels = labelIm) # N_lm, 2
        lm_coords_l = labelIm[lm_coords[:,0], lm_coords[:,1]] # N_lm
        # labels in labelIm are not necessarily continuous -> replace by rank
        label_rank = np.zeros(shape=(max(labels)+1,), dtype=np.int32)
        for l in labels:
            label_rank[l] = labels.index(l)
        lm_coords_l = label_rank[lm_coords_l]
        lm_coords_l = np.eye(len(labels))[lm_coords_l] # N_lm, N_ob
        lm_coords_ob = lm_coords[:,np.newaxis] * lm_coords_l[...,np.newaxis] # N_lm, N_ob, 1
        lm_coords_dist = np.sum(np.square(mass_centers - lm_coords_ob), 2, keepdims=True) # N_lm, N_ob, 1
        lm_coords_dinv= 1./(lm_coords_dist + 0.1) # N_lm, N_ob, 1
        lm_coords_dinv = lm_coords_dinv * lm_coords_ob # erase weights that are outside object
        wsum=np.sum(lm_coords_ob * lm_coords_dinv, 0, keepdims=False) # N_ob, 2
        sum=np.sum(lm_coords_dinv, 0, keepdims=False) # N_ob, 1
        centers = wsum / sum # N_ob, 2
    else:
        raise ValueError(f"Invalid center mode: {center_mode}")
    return dict(zip(labels, centers))

# channel dimension = frames
def _compute_prev_label_map(labelIm, prevlabelIm, end_points):
    n_chan = labelIm.shape[-1]
    for i in range(0, len(end_points)):
        for j in range(0, len(end_points[i])):
            if end_points[i][j]<0:
                end_points[i][j] = n_chan + end_points[i][j]
    min_frame=np.min(end_points)
    max_frame=np.max(end_points)
    assert min_frame<n_chan and max_frame<=n_chan, f"invalid end_points: min={min_frame}, max={max_frame}, nchan={n_chan}"
    labels_map_prev_by_c = dict()
    labels_by_c = {c : np.unique(labelIm[...,c]) for c in range(min_frame, max_frame+1)}
    for c in range(min_frame, max_frame+1):
        prev_labels = mean(prevlabelIm[..., c], labelIm[...,c], labels_by_c[c])
        labels_map_prev_by_c[c] = {label:int(prev_label) for label, prev_label in zip(labels_by_c[c], prev_labels)}
    labels_map_prev = []
    for (start, stop) in end_points:
        assert stop>=start, f"invalid endpoint [{start}, {stop}]"
        if start == stop: # same frame : prev labels = current label
            labels_map_prev.append({label:label for label in labels_by_c[stop]})
        elif start == stop-1: # successive frames: prev labels are simply those of prevlabelIm
            labels_map_prev.append(labels_map_prev_by_c[stop])
        else: # augmentation frame subsampling -> iterate through lineage to get the previous label @ last frame
            labels_map_prev_cur = labels_map_prev_by_c[stop]
            #print("endpoint lmp @ {} = {}".format(stop, labels_map_prev_))
            for c in range(stop-1, start, -1):
                labels_map_prev_temp = labels_map_prev_by_c[c]
                get_prev = lambda label : labels_map_prev_temp[label] if label in labels_map_prev_temp else 0
                labels_map_prev_cur = {label:get_prev(prev) for label,prev in labels_map_prev_cur.items()}
                #print("lmp @ {} = {}".format(c, labels_map_prev_cur))
            labels_map_prev.append(labels_map_prev_cur)
    return labels_map_prev

def _compute_displacement(labelIm, labels_map_prev, dyIm, dxIm, edm, center_mode, centerIm=None, centerImPrev=None, categoryIm=None, rankIm=None, rankImPrev=None, prevLabelArr=None):
    assert labelIm.shape[-1] == 2, f"invalid labelIm : {labelIm.shape[-1]} channels instead of 2"
    labels_map_centers = [_get_labels_and_centers(labelIm[...,c], edm[...,c], center_mode) for c in [0, 1]]
    if len(labels_map_centers[-1])==0: # no cells
        return
    curLabelIm = labelIm[...,-1]
    labels_prev = labels_map_centers[0].keys()
    labels_prev_rank = {l:r for r, l in enumerate(labels_prev)}
    for rank, (label, center) in enumerate(labels_map_centers[-1].items()):
        label_prev = labels_map_prev[label]
        if label_prev in labels_prev:
            dy = center[0] - labels_map_centers[0][label_prev][0] # axis 0 is y
            dx = center[1] - labels_map_centers[0][label_prev][1] # axis 1 is x
            if categoryIm is None and abs(dy)<1:
                dy = copysign(1, dy) # min value == 1 / same sign as dy
            if categoryIm is None and abs(dx)<1:
                dx = copysign(1, dx) # min value == 1 / same sign as dx
            mask = curLabelIm == label
            dyIm[mask] = dy
            dxIm[mask] = dx
            if rankIm is not None:
                rankIm[mask] = rank + 1
            if prevLabelArr is not None:
                prevLabelArr[rank] = labels_prev_rank[label_prev]+1
        elif rankIm is not None:
            rankIm[curLabelIm == label] = rank + 1
    if rankImPrev is not None:
        prevLabelIm = labelIm[...,0]
        for rank, label in enumerate(labels_map_centers[0].keys()):
            rankImPrev[prevLabelIm==label] = rank + 1
    if categoryIm is not None:
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
            categoryIm[curLabelIm == label] = value

    if centerIm is not None:
        assert centerIm.shape == dyIm.shape, "invalid shape for center image"
        _draw_centers(centerIm, labels_map_centers[-1], edm[...,1], labelIm[...,1])
    if centerImPrev is not None:
        assert centerImPrev.shape == dyIm.shape, "invalid shape for center image prev"
        _draw_centers(centerImPrev, labels_map_centers[0], edm[...,0], labelIm[...,0])
    # if noNextArr is not None:
    #     label_prev = labels_map_centers[0].keys()
    #     label_cur = labels_map_centers[-1].keys()
    #     label_cur_prev = {labels_map_prev[l] for l in label_cur if labels_map_prev[l]>0}
    #     no_next = label_prev - label_cur_prev
    #     for r,l in enumerate(label_prev):
    #         if l in no_next:
    #             noNextArr[r]=True

def _draw_centers(centerIm, labels_map_centers, edm, labelIm): # TODO design choice: draw gaussian and use L2 regression ?
    if len(labels_map_centers)==0:
        return

    # point
    for label, center in labels_map_centers.items(): # in case center prediction is a classification
        if isnan(center[0]) or isnan(center[1]):
            # print(f"Warning label: {label} nan center")
        else:
            drawn = False
            for dy,dx in itertools.product(range(2), range(2)):
                if int(center[0])==center[0]:
                    dy = 0
                if int(center[1])==center[1]:
                    dx = 0
                y = int(center[0])+dy
                x = int(center[1])+dx
                if labelIm[y, x] == label:
                    drawn = True
                    centerIm[y, x] = 1
                if not drawn: # center is outside.. draw it anyway
                    centerIm[int(center[0]+0.5), int(center[1]+0.5)] = 1
                    # print(f"Warning Label: {label} center {center} is outside")
    # gaussian
    # Y, X = centerIm.shape
    # Y, X = np.meshgrid(np.arange(Y, dtype = np.float32), np.arange(X, dtype = np.float32), indexing = 'ij')
    # sigmas = map_coordinates(edm, np.array(list(labels_map_centers.values())).T, prefilter=False)
    # for i, center in enumerate(centers):
    #     sigma_sq = max(1, 0.5 * (sigmas[i]**2))
    #     d = np.square(center[0] - Y) + np.square(center[1] - X)
    #     np.add(centerIm, np.exp(-d / sigma_sq), out=centerIm)
    # distance to center
    # Y, X = centerIm.shape
    # Y, X = np.meshgrid(np.arange(Y, dtype = np.float32), np.arange(X, dtype = np.float32), indexing = 'ij')
    # for i, (label, center) in enumerate(labels_map_centers.items()):
    #     mask = labelIm==label
    #     if mask.sum()>0:
    #         #w = -1# -1./max(1, 0.5 * (sigmas[i]**2))
    #         d = np.exp(-(np.square(Y-center[0])+np.square(X-center[1])))
    #         centerIm[mask] = d[mask]
