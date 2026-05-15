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

class DistnetIterator(TrackingIterator):
    def __init__(self,
                 dataset,
                 extract_tile_function,  # = extract_tile_random_zoom_function(tile_shape=(128, 128), n_tiles=8, zoom_range=[0.6, 1.6], aspect_ratio_range=[0.75, 1.5], random_channel_jitter_shape=[50, 50] ),
                 frame_window:int,
                 aug_frame_subsampling,  # either int: frame number interval will be drawn uniformly in [frame_window,aug_frame_subsampling] or callable that generate an frame number interval (int)
                 erase_edge_cell_size:int,
                 future_frames:bool = True,
                 allow_frame_subsampling_direct_neigh:bool = False,
                 aug_remove_prob: float = 0.005,
                 return_link_multiplicity:bool = True,
                 channel_keywords:list=CHANNEL_KEYWORDS,  # channel @1 must be label
                 input_label_keywords:list = None,  # additional labels that will be considered as input to the neural network
                 array_keywords:list=ARRAY_KEYWORDS[:1],  # if second array : category
                 elasticdeform_parameters:dict = None,
                 downscale_displacement_and_link_multiplicity=1,
                 return_edm_derivatives: bool = False,
                 return_center:bool = True,
                 scale_edm:bool = False,  # for each cell max edm value is 1
                 center_mode:str = "MEDOID",  # GEOMETRICAL, "EDM_MAX", "EDM_MEAN", "SKELETON", "MEDOID"
                 center_distance_mode = "GEODESIC",  # GEODESIC, EUCLIDEAN
                 input_label_center_idx:int = -1,  # center of target label are centers of the input label of this index
                 return_label_rank = False,
                 segmentation:bool = True,
                 tracking:bool = True,
                 long_term: bool = True,
                 return_next_displacement:bool = True,
                 frame_aware:bool=False,  # return actual frame (relative to central frame)
                 tridimensional_mode:bool=False,
                 z_radius:float=1.,  # to take into account anisotropy: z radius considering that x=y=1
                 return_loss_mask:bool=False,
                 category_keep_probabilities:list=None,  # per-category probability for balanced sampling. length = number of categories
                 **kwargs):
        assert len(channel_keywords)>=2, 'keyword should contain at least 2 elements in this order: grayscale input images, object labels, [other grayscale input images]'
        if frame_window == 0:
            tracking = False
        if tracking:
            assert 2 >= len(array_keywords) >= 1, 'array keyword first element should be links to previous objects. if 2 elements: second must be category'
        else:
            assert 1 >= len(array_keywords) >= 0, 'only one array keyword allow (category)'

        assert center_mode.upper() in CENTER_MODE, f"invalid center mode: {center_mode} should be in {CENTER_MODE}"
        assert center_distance_mode.upper() in CENTER_DISTANCE_MODE, f"invalid center distance mode: {center_distance_mode} should be in {CENTER_DISTANCE_MODE}"
        self.category_array_idx = next((i for i, s in enumerate(array_keywords) if "category" in s), -1)
        self.tracking = tracking
        self.segmentation=segmentation
        if not segmentation:
            return_center = False
            return_edm_derivatives = False
            scale_edm = False
        if not tracking:
            return_link_multiplicity = False
            return_next_displacement = False
            long_term = False
        if not segmentation and not tracking:
            assert self.category_array_idx==0, "if no segmentation and no tracking category must be provided"
        self.return_link_multiplicity=return_link_multiplicity
        self.downscale=downscale_displacement_and_link_multiplicity
        self.erase_edge_cell_size=erase_edge_cell_size
        self.aug_frame_subsampling=aug_frame_subsampling
        self.allow_frame_subsampling_direct_neigh=allow_frame_subsampling_direct_neigh
        self.return_edm_derivatives=return_edm_derivatives
        self.scale_edm = scale_edm
        self.return_center=return_center
        self.center_mode=center_mode.upper()
        self.center_distance_mode = center_distance_mode.upper()
        if input_label_center_idx >= 0:
            assert input_label_center_idx < len(input_label_keywords), f"invalid input_label_center_idx={input_label_center_idx} must be < len(input_label_keywords)={len(input_label_keywords)}"
        self.input_label_center_idx = input_label_center_idx
        self.return_label_rank=return_label_rank
        self.frame_window = frame_window
        self.return_next_displacement=return_next_displacement
        self.n_label_max = kwargs.pop("n_label_max", 2000)
        self.long_term=long_term
        self.frame_aware=frame_aware
        self.output_central_only = False
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
        self.z_radius = z_radius
        self.return_exclusion_weight_map = return_loss_mask
        self.category_keep_probabilities=category_keep_probabilities
        if return_loss_mask :
            print(f"Category keep probabilities: {category_keep_probabilities}")
        super().__init__(dataset=dataset,
                         channel_keywords=channel_keywords,
                         array_keywords = array_keywords,
                         input_channels=[0] + [i for i in range(2, nchan)] + self.label_input_channels,
                         output_channels=[1],
                         n_spatial_dims=3 if tridimensional_mode else 2,
                         channels_prev=[True]*len(channel_keywords),
                         channels_next=[future_frames] * len(channel_keywords),
                         mask_channels=[1] + self.label_input_channels,
                         n_frames = self.frame_window,
                         aug_remove_prob=aug_remove_prob,
                         frame_subsampling=1,  # disable default frame subsampling mechanism
                         aug_all_frames=False,
                         convert_masks_to_dtype=False,
                         return_image_index = self.frame_aware,
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

    def _get_batch_by_channel(self, index_array, perform_augmentation, input_only=False, perform_elasticdeform:bool=True, perform_tiling:bool=True, perform_channels_postprocessing:bool=True, **kwargs):
        if self.frame_window > 0:
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
        else:
            n_frames = 0
        #print(f"aug sub: {self.aug_frame_subsampling} -> nframes={n_frames}", flush=True)
        kwargs.update({"n_frames":n_frames, "ref_channel_idx":1 if not input_only else 0})
        if n_frames > 0: # only open subsampled frames, except for labels and prevlinks that requires all the frames to reconstruct lineage
            frames = np.array(self._get_end_points(n_frames, False)) - n_frames
            #print(f"frame increment: {frames}")
            kwargs.update({"frame_increment_per_channel": {c:frames for c in range(len(self.channel_keywords)) if c!=1 }})
            kwargs.update({"frame_increment_per_array": {self.category_array_idx : frames}})
        batch_by_channel, aug_param_array, ref_channel = super()._get_batch_by_channel(index_array, perform_augmentation, input_only, perform_elasticdeform=False, perform_tiling=False, perform_channels_postprocessing=False, **kwargs)
        ref_shape = batch_by_channel[0].shape
        for c in range(1, len(self.channel_keywords)):
            assert batch_by_channel[c].shape[:3] == ref_shape[:3], f"channel {c} shape is {batch_by_channel[c].shape} differs from channel 0: {ref_shape}"
        if not issubclass(batch_by_channel[1].dtype.type, np.integer): # label
            batch_by_channel[1] = batch_by_channel[1].astype(np.int32)
        # correction for oob @ previous labels : add identity links
        if self.tracking:
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
        if n_frames>1: # remove unused frames for labels
            sel = self._get_end_points(n_frames, False)
            #print(f"remove unused frames: nframes: {n_frames*2+1} sel: {sel} channels: {1}")
            batch_by_channel[1] = batch_by_channel[1][..., sel]
            if self.return_image_index:
                batch_by_channel["image_idx"] = batch_by_channel["image_idx"][:, sel]
        if perform_channels_postprocessing and self.channels_postprocessing_function is not None:
            self.channels_postprocessing_function(batch_by_channel)

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
            self._apply_tiling(batch_by_channel, index_array)
        if perform_elasticdeform or perform_tiling:
            for c in converted_from_float16:
                batch_by_channel[c] = batch_by_channel[c].astype('float16')
        if self.return_image_index: # reshape to match image shape
            n_chan = self.frame_window * (2 if self.channels_next[1] else 1) + 1
            spatial_axis = (1, 1, 1) if self.n_spatial_dims == 3 else (1, 1)
            batch_by_channel["image_idx"] = np.reshape(batch_by_channel["image_idx"], (-1,) + spatial_axis + (n_chan,))
        return batch_by_channel, aug_param_array, ref_channel

    @staticmethod
    def _set_identity_link(prev_label, b, c): # prev_label = (batch, label, [0=current label 1=prev label], channels)
        if np.all(prev_label[b, :, 0, c]==0): # fix: in case exported dataset has no label at first frame
            prev_label[b, :, 0, c] = np.arange(1, prev_label.shape[1]+1)
        prev_label[b, :, 1, c] = prev_label[b, :, 0, c]

    def _get_frames_to_augment(self, img, chan_idx, aug_params):
        if self.aug_all_frames or self.frame_window == 0:
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
        labels_map_prev = batch_by_channel["prev_label_map"] if self.tracking else None
        return_next = self.channels_next[1]
        long_term = self.long_term
        frame_window = self.frame_window
        if self.output_central_only and self.frame_window > 0:
            assert self.channels_prev[1], "in return_central_only mode previous must be returned"
            assert return_next, "in return_central_only mode next must be returned"
            for midx in self.mask_channels:
                batch_by_channel[midx] = batch_by_channel[midx][..., self.frame_window-1:self.frame_window+2] # only prev, central, and next frame
            labelIms = batch_by_channel[1]
            for aidx in range(len(batch_by_channel['arrays'])):
                batch_by_channel['arrays'][aidx] = batch_by_channel['arrays'][aidx][..., self.frame_window-1:self.frame_window+2]
            for icidx in range(len(batch_by_channel["input_centers"])): # remap : 0 -> fw-1, 1 -> fw, 2 -> fw+1
                centers = batch_by_channel["input_centers"][icidx]
                batch_by_channel["input_centers"][icidx] = {(b, c-(self.frame_window-1)):center for (b, c), center in centers.items() if self.frame_window-1<=c<=self.frame_window+1 }
            long_term = False
            frame_window = 1
            labels_map_prev = [lmp[self.frame_window-1:self.frame_window+2] for lmp in labels_map_prev] if self.tracking else None
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
        edm = np.zeros(shape=labelIms.shape, dtype=np.float32) if self.segmentation or ("edm" in self.center_mode and self.input_label_center_idx < 0) else None
        if edm is not None:
            for b,c in itertools.product(range(edm.shape[0]), range(edm.shape[-1])):
                edm[b,...,c] = edt_smooth(labelIms[b,...,c], object_slices[(b, c)], z_radius=self.z_radius)
                #edm[b,...,c] = edt.edt(labelIms[b,...,c], black_border=False, anisotropy=(self.z_radius, 1.0, 1.0) if self.z_radius is not None and self.z_radius != 1.0 else None)
        n_motion = 2 * frame_window if return_next else frame_window
        if long_term:
            n_motion = n_motion + (2 * ( frame_window - 1 ) if return_next else frame_window -1)
        dzIm = np.zeros(labelIms.shape[:-1] + (n_motion,), dtype=self.dtype) if self.tracking and self.n_spatial_dims == 3 else None
        dyIm = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype) if self.tracking else None
        dxIm = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype) if self.tracking else None
        if ndisp:
            dzImNext = np.zeros(labelIms.shape[:-1] + (n_motion,), dtype=self.dtype) if self.n_spatial_dims == 3 else None
            dyImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
            dxImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
            if self.return_link_multiplicity:
                linkMultiplicityImNext = np.zeros(labelIms.shape[:-1]+(n_motion,), dtype=self.dtype)
        centerIm = np.zeros(labelIms.shape, dtype=self.dtype) if self.return_center else None
        categoryIm = np.zeros(labelIms.shape, dtype=self.dtype) if self.category_array_idx>=0 else None
        cat_array = batch_by_channel['arrays'][self.category_array_idx] if self.category_array_idx>=0 else None
        if cat_array is not None:
            if self.n_spatial_dims == 3 and len(cat_array.shape) == 5:
                cat_array = cat_array[:, :, :, 0]
            elif not self.n_spatial_dims == 3 and len(cat_array.shape) == 4:
                cat_array = cat_array[:, :, 0]
        if self.return_label_rank:
            rankIm = np.zeros(labelIms.shape, dtype=np.int32)
            prevLabelArr = np.zeros(labelIms.shape[:1]+(n_motion, self.n_label_max), dtype=np.int32) if self.tracking else None
            nextLabelArr = np.zeros(labelIms.shape[:1] + (n_motion, self.n_label_max), dtype=np.int32) if self.tracking else None
            centerArr = np.zeros(labelIms.shape[:1]+labelIms.shape[-1:]+(self.n_label_max, self.n_spatial_dims), dtype=np.float32) # B, C, N, ndim
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
            input_centers = batch_by_channel["input_centers"][self.input_label_center_idx][(b, c)] if self.input_label_center_idx >= 0 else None
            labels_and_centers[(b, c)] = _get_labels_and_centers(labelIms[b][...,c], edm[b][...,c] if edm is not None else None, self.center_mode, input_centers=input_centers)
        for i in range(labelIms.shape[0]):
            bidx = get_idx(i)
            if self.frame_window > 0:
                for c in range(0, frame_window):
                    sel = [c, c+1]
                    l_c = [labels_and_centers[(i,s)] for s in sel]
                    o_s = [object_slices[(i, s)] for s in sel]
                    _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c] if self.tracking else None, o_s, dzIm=dzIm[i, ..., c] if dzIm is not None else None, dyIm=dyIm[i,...,c] if self.tracking else None, dxIm=dxIm[i,...,c] if self.tracking else None, dzImNext=dzImNext[i,...,c] if ndisp and self.n_spatial_dims == 3 else None, dyImNext=dyImNext[i,...,c] if ndisp else None, dxImNext=dxImNext[i,...,c] if ndisp else None, cdmIm=centerIm[i,...,frame_window] if self.return_center and sel[1] == frame_window else None, cdmImPrev=centerIm[i,...,c] if self.return_center else None, edmIm = edm[i,...,frame_window] if self.scale_edm else None, edmImPrev = edm[i,...,c] if self.scale_edm else None, scale_edm=self.scale_edm, categoryIm=categoryIm[i,...,frame_window] if self.category_array_idx>=0 and sel[1] == frame_window else None, categoryArray=cat_array[bidx, :, frame_window] if self.category_array_idx>=0 and sel[1] == frame_window else None, categoryImPrev=categoryIm[i,...,c] if self.category_array_idx>=0 else None, categoryArrayPrev=cat_array[bidx, :, c] if self.category_array_idx>=0 else None, linkMultiplicityIm=linkMultiplicityIm[i,...,c] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,...,c] if self.return_link_multiplicity and ndisp else None, rankIm=rankIm[i,...,frame_window] if self.return_label_rank and sel[1] == frame_window else None, rankImPrev=rankIm[i,...,c] if self.return_label_rank else None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank and self.tracking else None, nextLabelArr=nextLabelArr[i,c] if self.return_label_rank and self.tracking and ndisp else None, centerArr=centerArr[i,frame_window] if self.return_label_rank and sel[1] == frame_window else None, centerArrPrev=centerArr[i,c] if self.return_label_rank else None, center_distance_mode=self.center_distance_mode, z_radius=self.z_radius)
                if return_next:
                    for c in range(frame_window, 2*frame_window):
                        sel = [c, c+1]
                        l_c = [labels_and_centers[(i, s)] for s in sel]
                        o_s = [object_slices[(i, s)] for s in sel]
                        _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c] if self.tracking else None, o_s, dzIm=dzIm[i, ..., c] if dzIm is not None else None, dyIm=dyIm[i,...,c] if self.tracking else None, dxIm=dxIm[i,...,c] if self.tracking else None, dzImNext=dzImNext[i,...,c] if ndisp and self.n_spatial_dims == 3 else None, dyImNext=dyImNext[i,...,c] if ndisp else None, dxImNext=dxImNext[i,...,c] if ndisp else None, cdmIm=centerIm[i,..., c + 1] if self.return_center else None, edmIm = edm[i,...,c+1] if self.scale_edm else None, scale_edm=self.scale_edm, categoryIm=categoryIm[i,..., c + 1] if self.category_array_idx>=0 else None, categoryArray=cat_array[bidx, :, c+1] if self.category_array_idx>=0 else None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,...,c] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,...,c] if self.return_link_multiplicity and ndisp else None, rankIm=rankIm[i,..., c + 1] if self.return_label_rank else None, rankImPrev=None, prevLabelArr=prevLabelArr[i,c] if self.return_label_rank and self.tracking else None, nextLabelArr=nextLabelArr[i,c] if self.return_label_rank and ndisp else None, centerArr=centerArr[i, c + 1] if self.return_label_rank else None, center_distance_mode=self.center_distance_mode, z_radius=self.z_radius)
                if long_term:
                    off = 2*frame_window if return_next else frame_window
                    for c in range(0, frame_window-1):
                        sel = [c, frame_window]
                        l_c = [labels_and_centers[(i, s)] for s in sel]
                        o_s = [object_slices[(i, s)] for s in sel]
                        _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c + off], o_s, dzIm=dzIm[i, ..., c] if dzIm is not None else None, dyIm=dyIm[i,..., c + off], dxIm=dxIm[i,..., c + off], dzImNext=dzImNext[i,...,c] if ndisp and self.n_spatial_dims == 3 else None, dyImNext=dyImNext[i,..., c + off] if ndisp else None, dxImNext=dxImNext[i,..., c + off] if ndisp else None, cdmIm=None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,..., c + off] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,..., c + off] if self.return_link_multiplicity and ndisp else None, rankIm=None, rankImPrev=None, prevLabelArr=prevLabelArr[i, c + off] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i, c + off] if self.return_label_rank and ndisp else None, center_distance_mode=self.center_distance_mode, z_radius=self.z_radius)
                    if return_next:
                        for c in range(frame_window-1, 2*(frame_window-1)):
                            sel = [frame_window, c+3]
                            l_c = [labels_and_centers[(i, s)] for s in sel]
                            o_s = [object_slices[(i, s)] for s in sel]
                            _compute_outputs(l_c, labelIms[i][...,sel], labels_map_prev[bidx][c + off], o_s, dzIm=dzIm[i, ..., c] if dzIm is not None else None, dyIm=dyIm[i,..., c + off], dxIm=dxIm[i,..., c + off], dzImNext=dzImNext[i,...,c] if ndisp and self.n_spatial_dims == 3 else None, dyImNext=dyImNext[i,..., c + off] if ndisp else None, dxImNext=dxImNext[i,..., c + off] if ndisp else None, cdmIm=None, cdmImPrev=None, linkMultiplicityIm=linkMultiplicityIm[i,..., c + off] if self.return_link_multiplicity else None, linkMultiplicityImNext=linkMultiplicityImNext[i,..., c + off] if self.return_link_multiplicity and ndisp else None, rankIm=None, rankImPrev=None, prevLabelArr=prevLabelArr[i, c + off] if self.return_label_rank else None, nextLabelArr=nextLabelArr[i, c + off] if self.return_label_rank and ndisp else None, center_distance_mode=self.center_distance_mode, z_radius=self.z_radius)
            else:
                l_c = [labels_and_centers[(i, 0)]]
                o_s = [object_slices[(i, 0)]]
                _compute_outputs(l_c, labelIms[i][...,0:1], None, o_s, cdmIm=centerIm[i,...,0] if self.return_center else None, edmIm = edm[i,...,0] if self.scale_edm else None, scale_edm=self.scale_edm, categoryIm=categoryIm[i,...,0] if self.category_array_idx>=0 else None, categoryArray=cat_array[bidx, :, frame_window] if self.category_array_idx>=0 else None, rankIm=rankIm[i,...,0] if self.return_label_rank else None, centerArr=centerArr[i,0] if self.return_label_rank else None, center_distance_mode=self.center_distance_mode, z_radius=self.z_radius)

        if edm is not None:
            edm[edm == 0] = -1
        if self.return_edm_derivatives:
            der_y, der_x = np.zeros_like(edm), np.zeros_like(edm)
            if self.n_spatial_dims == 3:
                der_z = np.zeros_like(edm)
            c_range = range(edm.shape[-1]) if not self.output_central_only else range(frame_window, frame_window - 1)
            for b, c in itertools.product(range(edm.shape[0]), c_range):
                derivatives_labelwise(edm[b, ..., c], -1, der_z[b, ..., c] if self.n_spatial_dims == 3 else None, der_y[b, ..., c], der_x[b, ..., c], labelIms[b, ..., c],  object_slices[(b, c)])
            if self.output_central_only:
                der_y = der_y[..., 1:-1]
                der_x = der_x[..., 1:-1]
                if self.n_spatial_dims == 3:
                    der_z = der_z[..., 1:-1]

        if self.output_central_only: # select only central frame for edm / center and only displacement / link multiplicity related to central frame
            edm = edm[..., 1:-1] if edm is not None else None
            centerIm = centerIm[..., 1:-1] if self.return_center else None
            dyIm = dyIm[..., :1] if self.tracking else None
            dxIm = dxIm[..., :1] if self.tracking else None
            dzIm = dzIm[..., :1] if self.tracking and self.n_spatial_dims == 3 else None
            if self.return_link_multiplicity:
                linkMultiplicityIm = linkMultiplicityIm[..., :1]
            if self.category_array_idx>=0:
                categoryIm = categoryIm[..., 1:-1]
            if ndisp:
                dyImNext = dyImNext[..., 1:]
                dxImNext = dxImNext[..., 1:]
                if self.n_spatial_dims == 3:
                    dzImNext = dzImNext[..., 1:]
                if self.return_link_multiplicity:
                    linkMultiplicityImNext = linkMultiplicityImNext[..., 1:]
            if self.return_label_rank:
                rankIm = rankIm[..., 1:-1]
                centerArr = centerArr[: , 1:-1] # B, C, N, 2
                prevLabelArr = prevLabelArr[:, :1]  if self.tracking else None
                if ndisp:
                    nextLabelArr = nextLabelArr[:, 1:]
        if self.return_edm_derivatives:
            if self.n_spatial_dims == 3:
                edm = np.concatenate([edm, der_z, der_y, der_x], -1)
            else:
                edm = np.concatenate([edm, der_y, der_x], -1)
        all_channels = []
        if edm is not None:
            all_channels.append(edm)
        if self.return_center:
            all_channels.append(centerIm)
        downscale_factor = 1./self.downscale if self.downscale>1 else 0
        if self.n_spatial_dims == 3:
            scale = [1, 1, downscale_factor, downscale_factor, 1]  # don't downscale Z
        else:
            scale = [1, downscale_factor, downscale_factor, 1]
        if self.downscale>1 and self.tracking:
            dyIm = rescale(dyIm, scale, anti_aliasing= False, order=0)
            dxIm = rescale(dxIm, scale, anti_aliasing= False, order=0)
            if dzIm is not None:
                dzIm = rescale(dzIm, scale, anti_aliasing= False, order=0)
            if ndisp:
                dyImNext = rescale(dyImNext, scale, anti_aliasing= False, order=0)
                dxImNext = rescale(dxImNext, scale, anti_aliasing= False, order=0)
                if dzImNext is not None:
                    dzImNext = rescale(dzImNext, scale, anti_aliasing= False, order=0)
        if ndisp:
            dyIm = np.concatenate([dyIm, dyImNext], -1)
            dxIm = np.concatenate([dxIm, dxImNext], -1)
            if dzIm is not None:
                dzIm = np.concatenate([dzIm, dzImNext], -1)
        if self.tracking:
            if dzIm is not None:
                all_channels.append(dzIm)
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
        if self.category_array_idx>=0:
            if self.downscale > 1:
                categoryIm = rescale(categoryIm, scale, anti_aliasing=False, order=0)
            all_channels.append(categoryIm)
        if self.return_label_rank:
            all_channels.append(rankIm)
            if ndisp:
                prevLabelArr = np.concatenate([prevLabelArr, nextLabelArr], 1)
            if self.tracking:
                all_channels.append(prevLabelArr)
            all_channels.append(centerArr)
        if self.return_exclusion_weight_map:
            weight_map = np.ones(labelIms.shape, dtype=np.float32)
            if self.category_keep_probabilities is not None and self.category_array_idx >= 0:
                if self.tracking and labels_map_prev is not None:
                    # Tracking mode: exclude whole cell lines
                    for b in range(labelIms.shape[0]):
                        bidx = get_idx(b)
                        # Decide exclusions based on central frame
                        excluded_labels = {c: set() for c in range(labelIms.shape[-1])}
                        central_c = frame_window
                        cur_cat = cat_array[bidx, :, central_c]
                        for obj_idx, sl in enumerate(object_slices[(b, central_c)]):
                            if sl is not None:
                                cat = int(cur_cat[obj_idx])
                                if 0 <= cat < len(self.category_keep_probabilities):
                                    if np.random.random() >= self.category_keep_probabilities[cat]:
                                        excluded_labels[central_c].add(obj_idx + 1)
                        # Trace backward: labels_map_prev[bidx][c] maps labels in frame c+1 → frame c
                        for c in range(central_c - 1, -1, -1):
                            lmp = labels_map_prev[bidx][c]
                            for label in excluded_labels[c + 1]:
                                for prev_label in lmp.get(label, []):
                                    excluded_labels[c].add(prev_label)
                        # Trace forward: invert labels_map_prev to get frame c → frame c+1
                        n_total_frames = labelIms.shape[-1]
                        for c in range(central_c, n_total_frames - 1):
                            lmp = labels_map_prev[bidx][c]  # maps frame c+1 → frame c
                            fwd = {}
                            for next_label, prev_labels in lmp.items():
                                for pl in prev_labels:
                                    fwd.setdefault(pl, []).append(next_label)
                            for label in excluded_labels[c]:
                                for next_label in fwd.get(label, []):
                                    excluded_labels[c + 1].add(next_label)
                        # Apply exclusions to weight map
                        for c in range(n_total_frames):
                            for label in excluded_labels[c]:
                                mask = labelIms[b, ..., c] == label
                                weight_map[b, ..., c][mask] = 0
                else:
                    # Non-tracking mode: exclude per-frame independently
                    for b, c in itertools.product(range(labelIms.shape[0]), range(labelIms.shape[-1])):
                        bidx = get_idx(b)
                        cur_cat = cat_array[bidx, :, c] if cat_array is not None else None
                        if cur_cat is not None:
                            for obj_idx, sl in enumerate(object_slices[(b, c)]):
                                if sl is not None:
                                    cat = int(cur_cat[obj_idx])
                                    if 0 <= cat < len(self.category_keep_probabilities):
                                        if np.random.random() >= self.category_keep_probabilities[cat]:
                                            mask = labelIms[b, ..., c][sl] == obj_idx + 1
                                            weight_map[b, ..., c][sl][mask] = 0
            all_channels.append(weight_map)
        return all_channels

    def _get_input_batch(self, batch_by_channel, ref_chan_idx, aug_param_array):
        inputs = super()._get_input_batch(batch_by_channel, ref_chan_idx, aug_param_array)
        batch_by_channel["input_centers"] = []
        if len(self.label_input_channels)>0 : # for each input channel compute EDM and GCDM
            lidx = len(inputs) - len(self.label_input_channels)
            labels = inputs[lidx:]
            inputs = inputs[:lidx]
            for labelIms in labels: # compute EDM and GDCM for each additional input label image
                edm = np.zeros(shape=labelIms.shape, dtype=np.float32)
                gdcm = np.zeros(shape=labelIms.shape, dtype=np.float32)
                centers = dict()
                batch_by_channel["input_centers"].append(centers)
                for b, c in itertools.product(range(labelIms.shape[0]), range(labelIms.shape[-1])):
                    cur_labels = labelIms[b, ..., c].astype(np.int32)
                    object_slices = find_objects(cur_labels)
                    edm[b, ..., c] = edt_smooth(cur_labels, object_slices, z_radius=self.z_radius)
                    #edm[b, ..., c] = edt.edt(cur_labels, black_border=False, anisotropy=(self.z_radius, 1.0, 1.0) if self.z_radius is not None and self.z_radius != 1.0 else None)
                    labels_and_centers = _get_labels_and_centers(cur_labels, edm[b, ..., c], "MEDOID")
                    centers[(b, c)] = labels_and_centers.values() # record for output batch
                    _draw_centers(gdcm[b,...,c], labels_and_centers, cur_labels, object_slices, "GEODESIC", z_radius=self.z_radius)
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
    ndim = labelIm.ndim
    edge_labels = set()
    # only check Y and X edges (last 2 axes), not Z
    for axis in range(ndim - 2, ndim):
        slc_lo = [slice(None)] * ndim
        slc_hi = [slice(None)] * ndim
        slc_lo[axis] = 0
        slc_hi[axis] = -1
        edge_labels.update(np.unique(labelIm[tuple(slc_lo)].ravel()))
        edge_labels.update(np.unique(labelIm[tuple(slc_hi)].ravel()))
    edge_labels.discard(0)
    if len(edge_labels)>0:
        objects = find_objects(labelIm, max_label=max(edge_labels))
        return {idx+1 : sl for idx, sl in enumerate(objects) if sl is not None and idx+1 in edge_labels and np.sum(labelIm[sl] == idx+1)<=min_size}
    else:
        return dict()

# displacement computation utils

def _get_labels_and_centers(labelIm, edm, center_mode = "GEOMETRICAL", input_centers=None):
    labels = np.unique(labelIm)
    labels = [int(round(l)) for l in labels if l!=0]
    if len(labels)==0:
        return dict()
    if input_centers is not None: # map input center to labels and get other centers if necessary
        l_c = dict()
        for center in input_centers:
            l = labelIm[tuple(int(round(c)) for c in center)]
            if l>0:
                l_c[l] = center
        # remove labels with known centers
        labels = [l for l in labels if l not in l_c.keys()]
        if len(labels) == 0: # no unknown centers
            return l_c
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
        edm_lap = der.laplacian(gaussian_filter(edm, sigma=1.5))
        skeleton = edm_lap<0.25
        centers = [get_medoid(*np.asarray( (labelIm == l) & skeleton).nonzero()) for l in labels]
    elif center_mode == "MEDOID":
        centers = [get_medoid(*np.asarray(labelIm == l).nonzero()) for l in labels]
    else:
        raise ValueError(f"Invalid center mode: {center_mode}")
    new_l_c = dict(zip(labels, centers))
    if input_centers is None:
        return new_l_c
    else:
        l_c.update(new_l_c)
        return l_c

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
                    labels_map_gap_prev_by_f[f][prevlabelArray[i,0,f]] = (prevlabelArray[i,1,f], f-int(prevlabelArray[i,2,f])-1)
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

def _compute_outputs(labels_map_centers, labelIm, labels_map_prev, object_slices, dzIm=None, dyIm=None, dxIm=None, dzImNext=None, dyImNext=None, dxImNext=None, cdmIm=None, cdmImPrev=None, edmIm=None, edmImPrev=None, scale_edm:bool=False, categoryIm=None, categoryArray=None, categoryImPrev=None, categoryArrayPrev=None, linkMultiplicityIm=None, linkMultiplicityImNext=None, rankIm=None, rankImPrev=None, prevLabelArr=None, nextLabelArr=None, centerArr=None, centerArrPrev=None, center_distance_mode:str= "GEODESIC", z_radius:float=None):
    assert (dxImNext is None) == (dyImNext is None)
    curLabelIm = labelIm[...,-1]
    labels_prev = labels_map_centers[0].keys()
    labels_prev_rank = {l:r for r, l in enumerate(labels_prev)}
    labels_map_prev = _subset_label_map_prev(labels_map_prev, labels_prev, labels_map_centers[-1].keys()) if labels_map_prev is not None else None
    for rank, (label, center) in enumerate(labels_map_centers[-1].items()):
        label_prevs = labels_map_prev.get(label, []) if labels_map_prev is not None else []
        mask = curLabelIm == label
        if len(label_prevs)==1:
            label_prev = next(iter(label_prevs))
            center_prev = labels_map_centers[0][label_prev]
            dyIm[mask] = center[-2] - center_prev[-2]
            dxIm[mask] = center[-1] - center_prev[-1]
            if dzIm is not None:
                dzIm[mask] = center[0] - center_prev[0]
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
                dyImNext[mask] = center[-2] - center_next[-2]
                dxImNext[mask] = center[-1] - center_next[-1]
                if dzImNext is not None:
                    dzImNext[mask] = center[0] - center_next[0]
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
        assert cdmIm.shape == curLabelIm.shape, "invalid shape for center image"
        _draw_centers(cdmIm, labels_map_centers[-1], curLabelIm, object_slices[-1], center_distance_mode=center_distance_mode, z_radius=z_radius)
    if cdmImPrev is not None:
        assert cdmImPrev.shape == curLabelIm.shape, "invalid shape for center image prev"
        _draw_centers(cdmImPrev, labels_map_centers[0], labelIm[...,0], object_slices[0], center_distance_mode=center_distance_mode, z_radius=z_radius)
    if centerArr is not None:
        for rank, (label, center) in enumerate(labels_map_centers[-1].items()):
            for dim in range(len(center)):
                centerArr[rank, dim] = center[dim]
    if centerArrPrev is not None:
        for rank, (label, center) in enumerate(labels_map_centers[0].items()):
            for dim in range(len(center)):
                centerArrPrev[rank, dim] = center[dim]

def _center_has_nan(center):
    return any(isnan(c) for c in center)

def _draw_centers(centerIm, labels_map_centers, labelIm, object_slices, center_distance_mode:str, z_radius:float=None):
    if len(labels_map_centers)==0:
        return
    ndim = centerIm.ndim
    # geodesic distance to center
    if center_distance_mode == "GEODESIC":
        shape = centerIm.shape
        labelIm_dil = maximum_filter(labelIm, size=5)
        non_zero = labelIm>0
        labelIm_dil[non_zero] = labelIm[non_zero]
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                center = labels_map_centers.get(i+1)
                if not _center_has_nan(center):
                    sl = tuple( [slice(max(0, s.start - 2), min(s.stop + 2, ax), s.step) for s, ax in zip(sl, shape)])
                    mask = labelIm_dil[sl] == i + 1
                    m = np.ones_like(mask)
                    center_idx = tuple(int(round(center[dim])) - sl[dim].start for dim in range(ndim))
                    m[center_idx] = 0
                    m = ma.masked_array(m, ~mask)
                    if ndim == 3:
                        anisotropy = (z_radius, 1.0, 1.0) if z_radius is not None and z_radius != 1.0 else 1.0
                    else:
                        anisotropy = 1.0
                    centerIm[sl][mask] = skfmm.distance(m, dx=anisotropy)[mask]
    elif center_distance_mode == "EUCLIDEAN": # EDM to centers in the whole image space
        centers = []
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                center = labels_map_centers.get(i+1)
                if not _center_has_nan(center):
                    centers.append(list(center))
        compute_edm(centers, centerIm)
    else: # euclidean distance to center inside cells only
        grids = np.meshgrid(*[np.arange(s, dtype=np.float32) for s in centerIm.shape], indexing='ij')
        for i, (label, center) in enumerate(labels_map_centers.items()):
            if _center_has_nan(center):
                pass
            else:
                mask = labelIm==label
                if mask.sum()>0:
                    d2 = sum(np.square(g - c) for g, c in zip(grids, center))
                    if ndim == 3 and z_radius is not None and z_radius != 1.0:
                        # scale Z contribution by z_radius
                        d2 = d2 - np.square(grids[0] - center[0]) + np.square((grids[0] - center[0]) * z_radius)
                    centerIm[mask] = np.sqrt(d2)[mask]


def edt_smooth(labelIm, object_slices, z_radius=None):
    shape = labelIm.shape
    ndim = labelIm.ndim
    if ndim == 3:
        # Smooth only in YX: upsample and convolve each Z-slice independently
        Z = shape[0]
        yx_shape = shape[1:]
        upsample_kernel = (1, 2, 2)
        w = np.ones(shape=(3, 3), dtype=np.int8)
        threshold = w.size // 2
        upsampled = np.kron(labelIm, np.ones(upsample_kernel))
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                sl_z = sl[0]
                sl_yx = tuple([slice(max(s.start*2 - 1, 0), min(s.stop*2 + 1, ax*2), s.step) for s, ax in zip(sl[1:], yx_shape)])
                for z in range(sl_z.start, sl_z.stop):
                    sub_labelIm = upsampled[(z,) + sl_yx]
                    mask = sub_labelIm == i + 1
                    new_mask = convolve(mask.astype(np.int8), weights=w, mode="nearest") > threshold
                    sub_labelIm[mask] = 0
                    sub_labelIm[new_mask] = i + 1
        effective_z = (z_radius if z_radius is not None else 1.0) * 2  # Z not upsampled, YX upsampled 2x
        anisotropy = (effective_z, 1.0, 1.0)
        edm = edt.edt(upsampled, black_border=False, anisotropy=anisotropy)
        # downsample YX only by factor 2
        edm = edm.reshape(Z, yx_shape[0], 2, yx_shape[1], 2)
        edm = edm.mean(axis=(2, 4))
        edm = np.divide(edm, 2)
        edm[edm <= 0.25] = 0
    else:
        upsample_kernel = (2, 2)
        upsampled = np.kron(labelIm, np.ones(upsample_kernel))
        w = np.ones(shape=(3, 3), dtype=np.int8)
        threshold = w.size // 2
        for (i, sl) in enumerate(object_slices):
            if sl is not None:
                sl = tuple([slice(max(s.start*2 - 1, 0), min(s.stop*2 + 1, ax*2), s.step) for s, ax in zip(sl, shape)])
                sub_labelIm = upsampled[sl]
                mask = sub_labelIm == i + 1
                new_mask = convolve(mask.astype(np.int8), weights=w, mode="nearest") > threshold
                sub_labelIm[mask] = 0
                sub_labelIm[new_mask] = i + 1
        edm = edt.edt(upsampled, black_border=False)
        edm = edm.reshape(shape[0], 2, shape[1], 2)
        edm = edm.mean(axis=(1, 3))
        edm = np.divide(edm + 0.5, 2)
        edm[edm <= 0.5] = 0
    return edm


def derivatives_labelwise(image, bck_value, der_z, der_y, der_x, labelIm, object_slices):
    shape = labelIm.shape
    ndim = labelIm.ndim
    for (i, sl) in enumerate(object_slices):
        if sl is not None:
            sl = tuple([slice(max(s.start - 1, 0), min(s.stop + 1, ax), s.step) for s, ax in zip(sl, shape)])
            mask = labelIm[sl] == i + 1
            sub_im = np.copy(image[sl])
            sub_im[np.logical_not(mask)] = bck_value # erase neighboring cells
            sub_derivatives = der.der(sub_im, *range(ndim))
            if ndim == 3:
                sub_der_z, sub_der_y, sub_der_x = sub_derivatives
                if der_z is not None:
                    der_z[sl][mask] = sub_der_z[mask]
            else:
                sub_der_y, sub_der_x = sub_derivatives
            der_y[sl][mask] = sub_der_y[mask]
            der_x[sl][mask] = sub_der_x[mask]
    return der_y, der_x
