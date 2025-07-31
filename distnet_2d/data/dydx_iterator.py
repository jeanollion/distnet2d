from dataset_iterator import TrackingIterator
from dataset_iterator.tile_utils import extract_tile_random_zoom_function
import numpy as np
import numpy.ma as ma
from scipy.ndimage import center_of_mass, find_objects, maximum_filter, map_coordinates, gaussian_filter, convolve
from skimage.transform import rescale
from skimage.feature import peak_local_max
from scipy.spatial import distance
import skfmm
from math import copysign, isnan
import sys
import itertools
import edt
from random import random
from .center_edm import compute_edm
from .medoid import get_medoid
from ..utils import image_derivatives_np as der
from collections import deque
import time

CENTER_MODE = ["GEOMETRICAL", "EDM_MAX", "EDM_MEAN", "SKELETON", "MEDOID"]
CENTER_DISTANCE_MODE = ["GEODESIC", "EUCLIDEAN"]
CHANNEL_KEYWORDS = ['/raw', '/regionLabels']
ARRAY_KEYWORDS = ['/linksPrev', '/category']

class DyDxIterator(TrackingIterator):
    def __init__(self,
                 dataset,
                 extract_tile_function,  # = extract_tile_random_zoom_function(tile_shape=(128, 128), n_tiles=8, zoom_range=[0.6, 1.6], aspect_ratio_range=[0.75, 1.5], random_channel_jitter_shape=[50, 50] ),
                 frame_window:int,
                 aug_frame_subsampling,  # either int: frame number interval will be drawn uniformly in [frame_window,aug_frame_subsampling] or callable that generate an frame number interval (int)
                 erase_edge_cell_size:int,
                 next:bool = True,
                 allow_frame_subsampling_direct_neigh:bool = False,
                 aug_remove_prob: float = 0.005,
                 return_link_multiplicity:bool = True,
                 channel_keywords:list=CHANNEL_KEYWORDS,  # channel @1 must be label
                 input_label_keywords:list = None, # additional labels that will be considered as input to the neural network
                 array_keywords:list=ARRAY_KEYWORDS[:1], # if second array : category
                 elasticdeform_parameters:dict = None,
                 downscale_displacement_and_link_multiplicity=1,
                 return_edm_derivatives: bool = False,
                 return_center:bool = True,
                 scale_edm:bool = False, # for each cell max edm value is 1
                 center_mode:str = "MEDOID",  # GEOMETRICAL, "EDM_MAX", "EDM_MEAN", "SKELETON", "MEDOID"
                 center_distance_mode = "GEODESIC", # GEODESIC, EUCLIDEAN
                 return_label_rank = False,
                 long_term:bool = True,
                 return_next_displacement:bool = True,
                 output_float16=False,
                 **kwargs):
        assert len(channel_keywords)>=2, 'keyword should contain at least 2 elements in this order: grayscale input images, object labels, [other grayscale input images]'
        assert 2 >= len(array_keywords) >= 1, 'array keyword first element should be links to previous objects. if 2 elements: second must be cateogry'
        assert center_mode.upper() in CENTER_MODE, f"invalid center mode: {center_mode} should be in {CENTER_MODE}"
        assert center_distance_mode.upper() in CENTER_DISTANCE_MODE, f"invalid center distance mode: {center_distance_mode} should be in {CENTER_DISTANCE_MODE}"
        self.return_category = len(array_keywords)>1
        self.return_link_multiplicity=return_link_multiplicity
        self.downscale=downscale_displacement_and_link_multiplicity
        self.erase_edge_cell_size=erase_edge_cell_size
        self.aug_frame_subsampling=aug_frame_subsampling
        self.allow_frame_subsampling_direct_neigh=allow_frame_subsampling_direct_neigh
        self.output_float16=output_float16
        self.return_edm_derivatives=return_edm_derivatives
        self.scale_edm = scale_edm
        self.return_center=return_center
        self.center_mode=center_mode.upper()
        self.center_distance_mode = center_distance_mode.upper()
        self.return_label_rank=return_label_rank
        assert frame_window>=1, "frame_window must be >=1"
        self.frame_window = frame_window
        self.return_next_displacement=return_next_displacement
        self.n_label_max = kwargs.pop("n_label_max", 2000)
        self.long_term=long_term if self.frame_window>1 else False
        self.return_central_only = False
        nchan = len(channel_keywords)
        if input_label_keywords is not None:
            if not isinstance(input_label_keywords, list):
                input_label_keywords = list(input_label_keywords)
            channel_keywords = channel_keywords + input_label_keywords
            self.label_input_channels = [i + nchan for i in range(0, len(input_label_keywords))]
            image_data_generators = kwargs.get("image_data_generators")
            if len(image_data_generators) == nchan: # mask generators were not provided for input label channels
                for _ in range(len(input_label_keywords)): # append mask generator from label images
                    image_data_generators.append(image_data_generators[1])
        else:
            self.label_input_channels = []
        super().__init__(dataset=dataset,
                    channel_keywords=channel_keywords,
                    array_keywords = array_keywords,
                    input_channels=[0] + [i for i in range(2, nchan)] + self.label_input_channels,
                    output_channels=[1],
                    channels_prev=[True]*len(channel_keywords),
                    channels_next=[next]*len(channel_keywords),
                    mask_channels=[1] + self.label_input_channels,
                    n_frames = self.frame_window,
                    aug_remove_prob=aug_remove_prob,
                    aug_all_frames=False,
                    convert_masks_to_dtype=False,
                    extract_tile_function=extract_tile_function,
                    elasticdeform_parameters=elasticdeform_parameters,
                    **kwargs)

    def disable_random_transforms(self, data_augmentation:bool=True, channels_postprocessing:bool=False):
        params = super().disable_random_transforms(data_augmentation, channels_postprocessing)
        params["aug_frame_subsampling"] = self.aug_frame_subsampling
        self.aug_frame_subsampling = 1
        return params

    def enable_random_transforms(self, parameters):
        super().enable_random_transforms(parameters)
        if "aug_frame_subsampling" in parameters:
            self.aug_frame_subsampling = parameters["aug_frame_subsampling"]

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False, perform_elasticdeform=True, perform_tiling=True, **kwargs):
        if self.aug_remove_prob>0 and random() < self.aug_remove_prob:
            n_frames = 0 # flag that aug_remove = true
        else:
            if self.aug_frame_subsampling is not None :
                if callable(self.aug_frame_subsampling):
                    n_frames = self.aug_frame_subsampling()
                elif self.aug_frame_subsampling > self.frame_window:
                    n_frames = max(self.frame_window, np.random.randint(self.aug_frame_subsampling))
                else:
                    n_frames = self.frame_window
            else:
                n_frames = self.frame_window
        kwargs.update({"n_frames":n_frames})
        batch_by_channel, aug_param_array, ref_channel = super()._get_batch_by_channel(index_array, perform_augmentation, input_only, perform_elasticdeform=False, perform_tiling=False, **kwargs)
        ref_shape = batch_by_channel[0].shape
        for c in range(1, len(self.channel_keywords)):
            assert batch_by_channel[c].shape[:3] == ref_shape[:3], f"channel {c} shape is {batch_by_channel[c].shape} differs from channel 0: {ref_shape}"
        if not issubclass(batch_by_channel[1].dtype.type, np.integer): # label
            batch_by_channel[1] = batch_by_channel[1].astype(np.int32)
        # correction for oob @ previous labels : add identity links
        prevLinks = batch_by_channel['arrays'][0]
        for b in range(prevLinks.shape[0]):
            if n_frames > 0:
                for i in range(n_frames):
                    inc = n_frames - i
                    prev_inc = aug_param_array[b][ref_channel].get(f"prev_inc_{inc}", inc)
                    if prev_inc!=inc:
                        #print(f"oob prev: batch: {b} n_frames={n_frames}, frame_idx:{i} inc={inc} actual inc:{prev_inc} will replace at {i+1}")
                        self._set_identity_link(prevLinks, b, i+1)
                    if self.channels_next[1]:
                        next_inc = aug_param_array[b][ref_channel].get(f"next_inc_{inc}", inc)
                        if next_inc!=inc:
                            #print(f"oob next: batch: {b} n_frames={n_frames}, frame_idx:{i} inc={inc} actual inc:{next_inc} will replace prev labels at {n_frames+inc}")
                            self._set_identity_link(prevLinks, b, n_frames + inc)
            else: # n_frame == 0 ->
                for c in range(1, prevLinks.shape[-1]):
                    self._set_identity_link(prevLinks, b, c)
        # get previous labels and store in batch_by_channel BEFORE applying tiling and elastic deform
        self._get_prev_label(batch_by_channel, n_frames)
        batch_by_channel["batch_size"] = batch_by_channel[0].shape[0] # batch size is recorded here: it will be used in case of tiling
        if n_frames>1: # remove unused frames
            sel = self._get_end_points(n_frames, False)
            channels = [c for c in batch_by_channel if not isinstance(c, str) and c>=0]
            #print(f"remove unused frames: nframes: {n_frames*2+1} sel: {sel} channels: {channels}")
            for c in channels:
                batch_by_channel[c] = batch_by_channel[c][..., sel]
        if perform_elasticdeform or perform_tiling: ## elastic deform do not support float16 type -> temporarily convert to float32
            channels = [c for c in batch_by_channel.keys() if not isinstance(c, str) and c>=0]
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

    @staticmethod
    def _set_identity_link(prev_label, b, c): # prev_label = (batch, label, [0=current label 1=prev label], channels)
        if np.all(prev_label[b, :, 0, c]==0): # fix: in case exported dataset has no label at first frame
            prev_label[b, :, 0, c] = np.arange(1, prev_label.shape[1]+1)
        prev_label[b, :, 1, c] = prev_label[b, :, 0, c]

    def _get_frames_to_augment(self, img, chan_idx, aug_params):
        if self.aug_all_frames:
            return list(range(img.shape[-1]))
        n_frames = (img.shape[-1]-1)//2 if self.channels_prev[chan_idx] and self.channels_next[chan_idx] else img.shape[-1]-1
        return self._get_end_points(n_frames, False)

    def _get_end_points(self, n_frames, pairs):
        return_next = self.channels_next[1]

        if self.frame_window>1 and n_frames>2 and not self.allow_frame_subsampling_direct_neigh: # fix closest previous gap to -1
            end_points = [int( (n_frames-1) * i/(self.frame_window-1) + 0.5) for i in range(0, self.frame_window-1)] + [n_frames-1, n_frames]
        else:
            end_points = [int(n_frames * i/self.frame_window + 0.5) for i in range(0, self.frame_window+1)]
        if return_next:
            end_points = end_points + [2 * n_frames - e for e in end_points[:-1]][::-1]
        if pairs:
            end_point_pairs = [[end_points[i], end_points[i+1]] for i in range(0, self.frame_window)]
            if return_next:
                end_point_pairs = end_point_pairs + [[end_points[i+self.frame_window], end_points[i+self.frame_window+1]] for i in range(0, self.frame_window)]
            if self.long_term:
                end_point_pairs = end_point_pairs + [[end_points[i], end_points[self.frame_window]] for i in range(0, self.frame_window-1)]
                if return_next:
                    end_point_pairs = end_point_pairs + [[end_points[self.frame_window], end_points[i+self.frame_window+1]] for i in range(1, self.frame_window)]
            return end_point_pairs
        else:
            return end_points

    def _get_prev_label(self, batch_by_channel, n_frames):
        labelIms = batch_by_channel[1]
        prevlabelArrays = batch_by_channel['arrays'][0]
        return_next = self.channels_next[1]
        prev_label_map = []
        if n_frames <=0:
            n_frames = self.frame_window
        assert labelIms.shape[-1]==prevlabelArrays.shape[-1] and labelIms.shape[-1]==1+n_frames*(2 if return_next else 1), f"invalid channel number: labels: {labelIms.shape[-1]} prev labels: {prevlabelArrays.shape[-1]} n_frames: {n_frames}"
        end_point_pairs = self._get_end_points(n_frames, True)
        #print(f"n_frames: {n_frames}, nchan: {labelIms.shape[-1]}, frame_window: {self.frame_window}, return_next: {return_next}, end_points: {self._get_end_points(n_frames, False)}, end_point_pairs: {end_point_pairs}")
        for b in range(labelIms.shape[0]):
            prev_label_map.append(_compute_prev_label_map(labelIms[b], prevlabelArrays[b], end_point_pairs))
        batch_by_channel["prev_label_map"] = prev_label_map

    def _get_output_batch(self, batch_by_channel, ref_chan_idx, aug_param_array): # compute edm, center, dx & dy edm, link_multiplicity
        ndisp = self.return_next_displacement
        labelIms = batch_by_channel[1]
        labels_map_prev = batch_by_channel["prev_label_map"]
        return_next = self.channels_next[1]
        long_term = self.long_term
        frame_window = self.frame_window
        if self.return_central_only:
            assert self.channels_prev[1], "in return_central_only mode previous must be returned"
            assert return_next, "in return_central_only mode next must be returned"
            labelIms = labelIms[..., self.frame_window-1:self.frame_window+2] # only prev, central, and next frame
            long_term = False
            frame_window = 1
            labels_map_prev = [lmp[self.frame_window-1:self.frame_window+2] for lmp in labels_map_prev]
        # remove small object
        mask_to_erase_cur = [chan_idx for chan_idx in self.mask_channels if chan_idx!=1 and chan_idx in batch_by_channel]
        mask_to_erase_chan_cur = [frame_window if self.channels_prev[chan_idx] else 0 for chan_idx in mask_to_erase_cur]
        mask_to_erase_prev = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_prev[chan_idx]]
        mask_to_erase_chan_prev = [0] * len(mask_to_erase_prev)
        if return_next:
            mask_to_erase_next = [chan_idx for chan_idx in mask_to_erase_cur if self.channels_next[chan_idx]]
            mask_to_erase_chan_next = [frame_window+1 if self.channels_prev[chan_idx] else 1 for chan_idx in mask_to_erase_next]
        for i in range(labelIms.shape[0]):
            # cur timepoint
            self._erase_small_objects_at_edges(labelIms[i,...,frame_window], i, mask_to_erase_cur, mask_to_erase_chan_cur, batch_by_channel)
            # prev timepoints
            for j in range(0, frame_window):
                self._erase_small_objects_at_edges(labelIms[i,...,j], i, mask_to_erase_prev, [m+j for m in mask_to_erase_chan_prev], batch_by_channel)
            if return_next:
                for j in range(0, frame_window):
                    self._erase_small_objects_at_edges(labelIms[i,...,frame_window+1+j], i, mask_to_erase_next, [m+j for m in mask_to_erase_chan_next], batch_by_channel)
        object_slices = {}
        for b, c in itertools.product(range(labelIms.shape[0]), range(labelIms.shape[-1])):
            object_slices[(b, c)] = find_objects(labelIms[b,...,c])
        edm = np.zeros(shape=labelIms.shape, dtype=np.float32)
        for b,c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
            edm[b,...,c] = edt_smooth(labelIms[b,...,c], object_slices[(b, c)])
            #edm[b,...,c] = edt.edt(labelIms[b,...,c], black_border=False)
        n_motion = 2 * frame_window if return_next else frame_window
        if long_term:
            n_motion = n_motion + (2 * ( frame_window - 1 ) if return_next else frame_window -1)
        dyIm = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
        dxIm = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
        if ndisp:
            dyImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
            dxImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
            if self.return_link_multiplicity:
                linkMultiplicityImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
        centerIm = np.zeros(labelIms.shape, dtype=self.dtype) if self.return_center else None
        categoryIm = np.zeros(labelIms.shape, dtype=self.dtype) if self.return_category else None
        cat_array = batch_by_channel['arrays'][1] if self.return_category else None
        if cat_array is not None and len(cat_array.shape) == 4:
            cat_array = cat_array[:, :, 0]
        if self.return_label_rank:
            rankIm = np.zeros(labelIms.shape, dtype=np.int32)
            prevLabelArr = np.zeros(labelIms.shape[:1]+(n_motion, self.n_label_max), dtype=np.int32)
            nextLabelArr = np.zeros(labelIms.shape[:1] + (n_motion, self.n_label_max), dtype=np.int32)
            centerArr = np.zeros(labelIms.shape[:1]+labelIms.shape[-1:]+(self.n_label_max,2), dtype=np.float32)
            centerArr.fill(np.nan)
        if self.return_link_multiplicity:
            linkMultiplicityIm = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)

        batch_size = batch_by_channel["batch_size"]
        if labelIms.shape[0]>batch_size:
            #n_tiles = labelIms.shape[0]//batch_size
            #print("batch size: {}, n_tiles: {}".format(batch_size, n_tiles))
            get_idx = lambda x:x%batch_size # We assume here that the indexing order of tiling is tile x batch
        else:
            get_idx = lambda x:x
        labels_and_centers = {}
        for b,c in itertools.product(range(labelIms.shape[0]), range(labelIms.shape[-1])):
            labels_and_centers[(b, c)] = _get_labels_and_centers(labelIms[b][...,c], edm[b][...,c], self.center_mode)
        for i in range(labelIms.shape[0]):
            bidx = get_idx(i)
            for c in range(0, frame_window):
                sel = [c, c+1]
                l_c = [labels_and_centers[(i,s)] for s in sel]
                o_s = [object_slices[(i, s)] for s in sel]
                _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c], o_s, dyIm[i,...,c], dxIm[i,...,c], dyImNext=dyImNext[i,...,c] if ndisp else None, dxImNext=dxImNext[i,...,c] if ndisp else None, cdmIm=centerIm[i,...,frame_window] if self.return_center and sel[1] == frame_window else None, cdmImPrev=centerIm[i,...,c] if self.return_center else None, edmIm = edm[i,...,frame_window] if self.scale_edm else None, edmImPrev = edm[i,...,c] if self.scale_edm else None, scale_edm=self.scale_edm, categoryIm=categoryIm[i,...,frame_window] if self.return_category and sel[1] == frame_window else None, categoryArray=cat_array[bidx, :, frame_window] if self.return_category and sel[1] == frame_window else None, categoryImPrev=categoryIm[i,...,c] if self.return_category else None, categoryArrayPrev=cat_array[bidx, :, c] if self.return_category else None, linkMultiplicityIm=linkMultiplicityIm[i,...,c] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,...,c] if self.return_link_multiplicity and ndisp else None, rankIm=rankIm[i,...,frame_window] if self.return_label_rank and sel[1] == frame_window else None, rankImPrev=rankIm[i,...,c] if self.return_label_rank else None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i,c] if self.return_label_rank and ndisp else None, centerArr=centerArr[i,frame_window] if self.return_label_rank and sel[1] == frame_window else None, centerArrPrev=centerArr[i,c] if self.return_label_rank else None, center_distance_mode=self.center_distance_mode)
            if return_next:
                for c in range(frame_window, 2*frame_window):
                    sel = [c, c+1]
                    l_c = [labels_and_centers[(i, s)] for s in sel]
                    o_s = [object_slices[(i, s)] for s in sel]
                    _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c], o_s, dyIm[i,...,c], dxIm[i,...,c], dyImNext=dyImNext[i,...,c] if ndisp else None, dxImNext=dxImNext[i,...,c] if ndisp else None, cdmIm=centerIm[i,..., c + 1] if self.return_center else None, edmIm = edm[i,...,c+1] if self.scale_edm else None, scale_edm=self.scale_edm, categoryIm=categoryIm[i,..., c + 1] if self.return_category else None, categoryArray=cat_array[bidx, :, c+1] if self.return_category else None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,...,c] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,...,c] if self.return_link_multiplicity and ndisp else None, rankIm=rankIm[i,..., c + 1] if self.return_label_rank else None, rankImPrev=None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i,c] if self.return_label_rank and ndisp else None, centerArr=centerArr[i, c + 1] if self.return_label_rank else None, center_distance_mode=self.center_distance_mode)
            if long_term:
                off = 2*frame_window if return_next else frame_window
                for c in range(0, frame_window-1):
                    sel = [c, frame_window]
                    l_c = [labels_and_centers[(i, s)] for s in sel]
                    o_s = [object_slices[(i, s)] for s in sel]
                    _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c + off], o_s, dyIm[i,..., c + off], dxIm[i,..., c + off], dyImNext=dyImNext[i,..., c + off] if ndisp else None, dxImNext=dxImNext[i,..., c + off] if ndisp else None, cdmIm=None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,..., c + off] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,..., c + off] if self.return_link_multiplicity and ndisp else None, rankIm=None, rankImPrev=None, prevLabelArr=prevLabelArr[i, c + off] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i, c + off] if self.return_label_rank and ndisp else None, center_distance_mode=self.center_distance_mode)
                if return_next:
                    for c in range(frame_window-1, 2*(frame_window-1)):
                        sel = [frame_window, c+3]
                        l_c = [labels_and_centers[(i, s)] for s in sel]
                        o_s = [object_slices[(i, s)] for s in sel]
                        _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c + off], o_s, dyIm[i,..., c + off], dxIm[i,..., c + off], dyImNext=dyImNext[i,..., c + off] if ndisp else None, dxImNext=dxImNext[i,..., c + off] if ndisp else None, cdmIm=None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,..., c + off] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,..., c + off] if self.return_link_multiplicity and ndisp else None, rankIm=None, rankImPrev=None, prevLabelArr=prevLabelArr[i, c + off] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i, c + off] if self.return_label_rank and ndisp else None, center_distance_mode=self.center_distance_mode)

        edm[edm == 0] = -1
        if self.return_edm_derivatives:
            der_y, der_x = np.zeros_like(edm), np.zeros_like(edm)
            for b, c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
                derivatives_labelwise(edm[b, ..., c], -1, der_y[b, ..., c], der_x[b, ..., c], labelIms[b, ..., c],  object_slices[(b, c)])
            if self.return_central_only:
                der_y = der_y[..., frame_window:frame_window+1]
                der_x = der_x[..., frame_window:frame_window+1]

        if self.return_central_only: # select only central frame for edm / center and only displacement / link multiplicity related to central frame
            edm = edm[..., frame_window:frame_window+1]
            centerIm = centerIm[..., frame_window:frame_window+1]
            dyIm = dyIm[..., frame_window-1:frame_window]
            dxIm = dxIm[..., frame_window-1:frame_window]
            if self.return_link_multiplicity:
                linkMultiplicityIm = linkMultiplicityIm[..., frame_window-1:frame_window]
            if self.return_category:
                categoryIm = categoryIm[..., frame_window:frame_window+1]
            if ndisp:
                dyImNext = dyImNext[..., frame_window-1:frame_window]
                dxImNext = dxImNext[..., frame_window-1:frame_window]
                if self.return_link_multiplicity:
                    linkMultiplicityImNext = linkMultiplicityImNext[..., frame_window-1:frame_window]
            if self.return_label_rank:
                rankIm = rankIm[..., frame_window:frame_window+1]
                centerArr = centerArr[: , frame_window:frame_window+1]
                prevLabelArr = prevLabelArr[:, :1]
                if ndisp:
                    nextLabelArr = nextLabelArr[:, 1:]
        if self.return_edm_derivatives:
            edm = np.concatenate([edm, der_y, der_x], -1)
        if self.output_float16:
            edm = edm.astype('float16', copy=False)
        all_channels = [edm]
        if self.return_center:
            if self.output_float16:
                centerIm = centerIm.astype('float16', copy=False)
            all_channels.append(centerIm)
        downscale_factor = 1./self.downscale if self.downscale>1 else 0
        scale = [1, downscale_factor, downscale_factor, 1]
        if self.downscale>1:
            dyIm = rescale(dyIm, scale, anti_aliasing= False, order=0)
            dxIm = rescale(dxIm, scale, anti_aliasing= False, order=0)
            if ndisp:
                dyImNext = rescale(dyImNext, scale, anti_aliasing= False, order=0)
                dxImNext = rescale(dxImNext, scale, anti_aliasing= False, order=0)
        if ndisp:
            dyIm = np.concatenate([dyIm, dyImNext], -1)
            dxIm = np.concatenate([dxIm, dxImNext], -1)
        if self.output_float16:
            dyIm = dyIm.astype('float16', copy=False)
            dxIm = dxIm.astype('float16', copy=False)
        all_channels.append(dyIm)
        all_channels.append(dxIm)
        if self.return_link_multiplicity:
            if self.downscale>1:
                linkMultiplicityIm = rescale(linkMultiplicityIm, scale, anti_aliasing= False, order=0)
                if ndisp:
                    linkMultiplicityImNext = rescale(linkMultiplicityImNext, scale, anti_aliasing= False, order=0)
            if ndisp:
                linkMultiplicityIm = np.concatenate([linkMultiplicityIm, linkMultiplicityImNext], -1)
            all_channels.append(linkMultiplicityIm)
        if self.return_category:
            if self.downscale > 1:
                categoryIm = rescale(categoryIm, scale, anti_aliasing=False, order=0)
            all_channels.append(categoryIm)
        if self.return_label_rank:
            if ndisp:
                prevLabelArr = np.concatenate([prevLabelArr, nextLabelArr], 1)
            all_channels.append(rankIm)
            all_channels.append(prevLabelArr)
            all_channels.append(centerArr)
        return all_channels

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        inputs = super()._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
        if len(self.label_input_channels)>0 : # for each input channel compute EDM and GCDM
            lidx = len(inputs) - len(self.label_input_channels)
            labels = inputs[lidx:]
            inputs = inputs[:lidx]
            for labelIms in labels: # compute EDM and GDCM for each additional input label image
                edm = np.zeros(shape=labelIms.shape, dtype=np.float32)
                gdcm = np.zeros(shape=labelIms.shape, dtype=np.float32)
                for b, c in itertools.product(range(labelIms.shape[0]), range(labelIms.shape[-1])):
                    cur_labels = labelIms[b, ..., c].astype(np.int32)
                    object_slices = find_objects(cur_labels)
                    edm[b, ..., c] = edt_smooth(cur_labels, object_slices)
                    labels_and_centers = _get_labels_and_centers(cur_labels, edm[b, ..., c], "MEDOID")
                    _draw_centers(gdcm[b,...,c], labels_and_centers, cur_labels, object_slices, "GEODESIC")
                inputs.append(edm)
                inputs.append(gdcm)
        return inputs

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
            edm_label = ma.array(edm, mask=labelIm != label)
            center = ma.argmax(edm_label, fill_value=0)
            center = np.unravel_index(center, edm_label.shape)
            centers.append(center)
    elif center_mode == "EDM_MEAN":
        assert edm is not None and edm.shape == labelIm.shape
        centers = center_of_mass(edm, labelIm, labels)
    elif center_mode == "SKELETON":
        edm_lap = der.laplacian_2d(gaussian_filter(edm, sigma=1.5))
        skeleton = edm_lap<0.25
        centers = [get_medoid(*np.asarray( (labelIm == l) & skeleton).nonzero()) for l in labels]
    elif center_mode == "MEDOID":
        centers = [get_medoid(*np.asarray(labelIm == l).nonzero()) for l in labels]
    else:
        raise ValueError(f"Invalid center mode: {center_mode}")
    return dict(zip(labels, centers))

# channel dimension = frames
def _compute_prev_label_map(labelIm, prevlabelArray, end_points):
    n_chan = labelIm.shape[-1]
    for i in range(0, len(end_points)):
        for j in range(0, len(end_points[i])):
            if end_points[i][j]<0:
                end_points[i][j] = n_chan + end_points[i][j]
    min_frame=np.min(end_points)
    max_frame=np.max(end_points)
    assert min_frame<n_chan and max_frame<=n_chan, f"invalid end_points: min={min_frame}, max={max_frame}, nchan={n_chan}"
    labels_map_prev_by_f = dict() # for each frame: label map set of adjacent prev labels
    labels_map_gap_prev_by_f = dict() # for each frame: label map tuple[prev label, prev frame]
    labels_by_f = {f : np.unique(labelIm[...,f]) for f in range(min_frame, max_frame+1)}
    for f in range(min_frame, max_frame+1):
        labels_map_prev_by_f[f] = dict()
        labels_map_gap_prev_by_f[f] = dict()
        for i in range(prevlabelArray.shape[0]):
            if prevlabelArray[i,0,f]>0 and prevlabelArray[i,1,f]>0:
                if prevlabelArray.shape[1]==2 or prevlabelArray[i,2,f]==0: # no gap
                    if prevlabelArray[i,0,f] not in labels_map_prev_by_f[f]:
                        labels_map_prev_by_f[f][prevlabelArray[i,0,f]] = {prevlabelArray[i,1,f]}
                    else:
                        labels_map_prev_by_f[f][prevlabelArray[i,0,f]].add(prevlabelArray[i,1,f])
                else: # gap
                    labels_map_gap_prev_by_f[f][prevlabelArray[i,0,f]] = (prevlabelArray[i,1,f], f-prevlabelArray[i,2,f]-1)
    labels_map_prev = []
    for (start, stop) in end_points:
        assert stop>=start, f"invalid endpoint [{start}, {stop}]"
        if start == stop: # same frame : prev labels = current label
            labels_map_prev.append({label:{label} for label in labels_by_f[stop]})
        elif start == stop-1: # successive frames: prev labels are simply those of prevlabelArray with zero gap
            labels_map_prev.append(labels_map_prev_by_f[stop])
        else: # non-successive frames iterate through lineage to get the previous label @ start frame
            labels_map_prev_cur = {l : _get_labels_at_frame(l, stop, start, labels_map_prev_by_f, labels_map_gap_prev_by_f) for l in labels_by_f[stop]}
            labels_map_prev.append(labels_map_prev_cur)
    return labels_map_prev

def _get_labels_at_frame(label, frame, target_previous_frame, labels_map_prev_by_f, labels_map_gap_prev_by_f):
    if frame < target_previous_frame:
        return []
    if frame == target_previous_frame:
        return [label]

    visited = set()
    queue = deque()
    queue.append((label, frame))
    visited.add((label, frame))
    result = set()

    while queue:
        current_label, current_frame = queue.popleft()
        if current_frame == target_previous_frame:
            result.add(current_label)
            continue
        if current_frame < target_previous_frame:
            continue  # Skip if we've gone past the target frame
        # Check gap links first (mutually exclusive with prev frame links)
        if current_frame in labels_map_gap_prev_by_f and current_label in labels_map_gap_prev_by_f[current_frame]:
            lf = labels_map_gap_prev_by_f[current_frame][current_label]
            if lf[1] >= target_previous_frame and lf not in visited:
                visited.add(lf)
                queue.append(lf)
        else: # Check immediate previous frame links
            if current_frame in labels_map_prev_by_f and current_label in labels_map_prev_by_f[current_frame]:
                for prev_label in labels_map_prev_by_f[current_frame][current_label]:
                    lf = (prev_label, current_frame - 1)
                    if lf not in visited:
                        visited.add(lf)
                        queue.append(lf)
    return set(result)

def _labels_map_prev_to_next(labels_map_prev):
    res = dict()
    for l, prevs in labels_map_prev.items():
        for p in prevs:
            if p in res:
                res[p].add(l)
            else:
                res[p] = {l}
    return res

def _subset_label_map_prev(labels_map_prev, prev_labels, labels):
    res = dict()
    for label in labels:
        if label in labels_map_prev:
            prevs = labels_map_prev[label]
            if len(prevs)>0:
                prevs_sub ={p for p in prevs if p in prev_labels}
                if len(prevs_sub)>0:
                    res[label] = prevs_sub
    return res

def _get_link_multiplicity(n_neigh):
    if n_neigh == 0:
        return 3
    elif n_neigh == 1:
        return 1
    else:
        return 2

def _compute_outputs(labels_map_centers, labelIm, labels_map_prev, object_slices, dyIm, dxIm, dyImNext=None, dxImNext=None, cdmIm=None, cdmImPrev=None, edmIm=None, edmImPrev=None, scale_edm:bool=False, categoryIm=None, categoryArray=None, categoryImPrev=None, categoryArrayPrev=None, linkMultiplicityIm=None, linkMultiplicityImNext=None, rankIm=None, rankImPrev=None, prevLabelArr=None, nextLabelArr=None, centerArr=None, centerArrPrev=None, center_distance_mode:str= "GEODESIC"):
    assert labelIm.shape[-1] == 2, f"invalid labelIm : {labelIm.shape[-1]} channels instead of 2"
    assert (dxImNext is None) == (dyImNext is None)
    curLabelIm = labelIm[...,-1]
    labels_prev = labels_map_centers[0].keys()
    labels_prev_rank = {l:r for r, l in enumerate(labels_prev)}
    labels_map_prev = _subset_label_map_prev(labels_map_prev, labels_prev, labels_map_centers[-1].keys())
    for rank, (label, center) in enumerate(labels_map_centers[-1].items()):
        label_prevs = labels_map_prev.get(label, [])
        mask = curLabelIm == label
        if len(label_prevs)==1:
            label_prev = next(iter(label_prevs))
            center_prev = labels_map_centers[0][label_prev]
            dy = center[0] - center_prev[0] # axis 0 is y
            dx = center[1] - center_prev[1] # axis 1 is x
            dyIm[mask] = dy
            dxIm[mask] = dx
            if prevLabelArr is not None:
                prevLabelArr[rank] = labels_prev_rank[label_prev]+1
        if linkMultiplicityIm is not None:
            linkMultiplicityIm[mask] = _get_link_multiplicity(len(label_prevs))
        if categoryIm is not None:
            categoryIm[mask] = categoryArray[label - 1] + 1
        if scale_edm and edmIm is not None:
            edmIm_masked = edmIm[mask]
            norm = np.max(edmIm_masked)
            if norm > 0:
                edmIm[mask] = edmIm_masked / norm
        if rankIm is not None:
            rankIm[mask] = rank + 1
    if dyImNext is not None:
        labels_next_rank = {l:r for r, l in enumerate(labels_map_centers[-1].keys())}
        labels_map_next = _labels_map_prev_to_next(labels_map_prev)
        for rank, (label, center) in enumerate(labels_map_centers[0].items()):
            label_nexts = labels_map_next.get(label, [])
            mask = labelIm[...,0] == label
            if len(label_nexts)==1:
                label_next = next(iter(label_nexts))
                center_next = labels_map_centers[-1][label_next]
                dy = center[0] - center_next[0] # axis 0 is y
                dx = center[1] - center_next[1] # axis 1 is x
                dyImNext[mask] = dy
                dxImNext[mask] = dx
                if nextLabelArr is not None:
                    nextLabelArr[rank] = labels_next_rank[label_next]+1
            if linkMultiplicityImNext is not None:
                linkMultiplicityImNext[mask] = _get_link_multiplicity(len(label_nexts))
            if categoryImPrev is not None:
                categoryImPrev[mask] = categoryArrayPrev[label - 1] + 1
            if scale_edm and edmImPrev is not None:
                edmIm_masked = edmImPrev[mask]
                norm = np.max(edmIm_masked)
                if norm > 0:
                    edmImPrev[mask] = edmIm_masked / norm
            if rankImPrev is not None:
                rankImPrev[mask] = rank + 1
    if cdmIm is not None:
        assert cdmIm.shape == dyIm.shape, "invalid shape for center image"
        _draw_centers(cdmIm, labels_map_centers[-1], labelIm[...,1], object_slices[1], center_distance_mode=center_distance_mode)
    if cdmImPrev is not None:
        assert cdmImPrev.shape == dyIm.shape, "invalid shape for center image prev"
        _draw_centers(cdmImPrev, labels_map_centers[0], labelIm[...,0], object_slices[0], center_distance_mode=center_distance_mode)
    if centerArr is not None:
        for rank, (label, center) in enumerate(labels_map_centers[-1].items()):
            centerArr[rank,0] = center[0]
            centerArr[rank,1] = center[1]
    if centerArrPrev is not None:
        for rank, (label, center) in enumerate(labels_map_centers[0].items()):
            centerArrPrev[rank,0] = center[0]
            centerArrPrev[rank,1] = center[1]

def _draw_centers(centerIm, labels_map_centers, labelIm, object_slices, center_distance_mode:str):
    if len(labels_map_centers)==0:
        return
    # geodesic distance to center
    if center_distance_mode == "GEODESIC":
        shape = centerIm.shape
        labelIm_dil = maximum_filter(labelIm, size=5)
        non_zero = labelIm>0
        labelIm_dil[non_zero] = labelIm[non_zero]
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                center = labels_map_centers.get(i+1)
                if not (isnan(center[0]) or isnan(center[1])):
                    sl = tuple( [slice(max(0, s.start - 2), min(s.stop + 2, ax), s.step) for s, ax in zip(sl, shape)])
                    mask = labelIm_dil[sl] == i + 1
                    m = np.ones_like(mask)
                    #print(f"label: {i+1} slice: {sl}, center: {center}, sub_m {sub_m.shape}, coord: {(int(round(center[0]))-sl[0].start, int(round(center[1]))-sl[1].start)}", flush=True)
                    m[int(round(center[0]))-sl[0].start, int(round(center[1]))-sl[1].start] = 0
                    m = ma.masked_array(m, ~mask)
                    centerIm[sl][mask] = skfmm.distance(m)[mask]
    elif center_distance_mode == "EUCLIDEAN": # EDM to centers in the whole image space
        centers = []
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                center = labels_map_centers.get(i+1)
                if not (isnan(center[0]) or isnan(center[1])):
                    centers.append([center[0], center[1]])
        compute_edm(centers, centerIm)
    else: # euclidean distance to center inside cells only
        Y, X = centerIm.shape
        Y, X = np.meshgrid(np.arange(Y, dtype = np.float32), np.arange(X, dtype = np.float32), indexing = 'ij')
        for i, (label, center) in enumerate(labels_map_centers.items()): # in case center prediction is a classification
            if isnan(center[0]) or isnan(center[1]):
                pass
            else:
                # distance to center
                mask = labelIm==label
                if mask.sum()>0:
                    d = np.sqrt(np.square(Y-center[0])+np.square(X-center[1]))
                    centerIm[mask] = d[mask]


def edt_smooth(labelIm, object_slices):
    shape = labelIm.shape
    upsampled = np.kron(labelIm, np.ones((2, 2))) # upsample by factor 2
    w=np.ones(shape=(3, 3), dtype=np.int8)
    for (i, sl) in enumerate(object_slices):
        if sl is not None:
            sl = tuple([slice(max(s.start*2 - 1, 0), min(s.stop*2 + 1, ax*2), s.step) for s, ax in zip(sl, shape)])
            sub_labelIm = upsampled[sl]
            mask = sub_labelIm == i + 1
            new_mask = convolve(mask.astype(np.int8), weights=w, mode="nearest") > 4 # smooth borders
            sub_labelIm[mask] = 0  # replace mask by smoothed
            sub_labelIm[new_mask] = i + 1
    edm = edt.edt(upsampled)
    edm = edm.reshape((shape[0], 2, shape[1], 2)).mean(-1).mean(1) # downsample (bin) by factor 2
    edm = np.divide(edm + 0.5, 2)  #convert to pixel unit
    edm[edm <= 0.5] = 0
    return edm


def derivatives_labelwise(image, bck_value, der_y, der_x, labelIm, object_slices):
    shape = labelIm.shape
    for (i, sl) in enumerate(object_slices):
        if sl is not None:
            sl = tuple([slice(max(s.start - 1, 0), min(s.stop + 1, ax), s.step) for s, ax in zip(sl, shape)])
            mask = labelIm[sl] == i + 1
            sub_im = np.copy(image[sl])
            sub_im[np.logical_not(mask)] = bck_value # erase neighboring cells
            sub_der_y, sub_der_x = der.der_2d(sub_im, 0, 1)
            der_y[sl][mask] = sub_der_y[mask]
            der_x[sl][mask] = sub_der_x[mask]
    return der_y, der_x
