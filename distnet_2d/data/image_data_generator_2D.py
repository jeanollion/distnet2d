from keras_preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator # this version doesn't have interpolation_order
import numpy as np
from math import tan, atan, pi, copysign
from . import pre_processing as pp
from random import getrandbits, uniform, choice
import copy
import scipy.ndimage as ndi

class ImageDataGenerator2D(ImageDataGenerator):
    """Short summary.

    Parameters
    ----------
    rotate90 : bool
        Description of parameter `rotate90`.
    perform_illumination_augmentation : bool
        Description of parameter `perform_illumination_augmentation`.
    gaussian_blur_range : list
        Description of parameter `gaussian_blur_range`.
    noise_intensity : float
        Description of parameter `noise_intensity`.
    histogram_scaling_mode : str
        PHASE_CONTRAST: per image scaling to random range in [min, max] with max in [0, 1] and min in [0, 1] and max-min > min_histogram_range
        RANDOM_MIN_MAX : per image scaling, mapping a random centile in min_centile_range to 0 and a random centile in max_centile_range to 1
        FLUORESCENCE: per image scaling to random center c and scale s with: c in histogram_normalization_center and s in histogram_normalization_scale ( I = (I - c) / s)
        TRANSMITTED_LIGHT: per image scaling to random center c and scale s with: c in [mean + histogram_normalization_center[0], mean + histogram_normalization_center[1]] and s in [sd * histogram_normalization_scale[0], sd * histogram_normalization_scale[1]] and mean = mean(I) sd = sd(I) : I = (I - c) / s
        AUTO: PHAST_CONTRAST if histogram_normalization_center is None or histogram_normalization_center is None else FLUORESCENCE
        NONE: no scaling
    min_histogram_range : float
        Description of parameter `min_histogram_range`.
    min_histogram_to_zero : bool
        Description of parameter `min_histogram_to_zero`.
    histogram_normalization_center : list
        Description of parameter `histogram_normalization_center`.
    histogram_normalization_scale : list
        Description of parameter `histogram_normalization_scale`.
    histogram_voodoo_n_points : int
        Description of parameter `histogram_voodoo_n_points`.
    histogram_voodoo_intensity : float
        Description of parameter `histogram_voodoo_intensity`.
    illumination_voodoo_n_points : int
        Description of parameter `illumination_voodoo_n_points`.
    illumination_voodoo_intensity : float
        Description of parameter `illumination_voodoo_intensity`.
    **kwargs : type
        Description of parameter `**kwargs`.

    Attributes
    ----------
    rotate90
    min_histogram_range
    min_histogram_to_zero
    noise_intensity
    gaussian_blur_range
    histogram_voodoo_n_points
    histogram_voodoo_intensity
    illumination_voodoo_n_points
    illumination_voodoo_intensity
    perform_illumination_augmentation
    histogram_normalization_center
    histogram_normalization_scale

    """
    def __init__(self, rotate90:bool=False, interpolation_order=1, perform_illumination_augmentation:bool = True, gaussian_blur_range:list=[1, 2], noise_intensity:float = 0.1, histogram_scaling_mode:str="AUTO", min_centile_range = [0, 5], max_centile_range=[95, 100], min_histogram_range:float=0.1, min_histogram_to_zero:bool=False, invert:bool=False, histogram_normalization_center=None, histogram_normalization_scale=None, histogram_voodoo_n_points:int=5, histogram_voodoo_intensity:float=0.5, illumination_voodoo_n_points:int=5, illumination_voodoo_intensity:float=0.6, **kwargs):
        assert histogram_scaling_mode in ["PHASE_CONTRAST", "RANDOM_MIN_MAX", "FLUORESCENCE", "TRANSMITTED_LIGHT", "AUTO", "NONE"], "invalid histogram scaling mode"
        if histogram_scaling_mode=="FLUORESCENCE" or histogram_scaling_mode=="TRANSMITTED_LIGHT" or (histogram_scaling_mode=="AUTO" and histogram_normalization_center is not None and histogram_normalization_scale is not None):
            assert histogram_normalization_center is not None and histogram_normalization_scale is not None, "in FLUORESCENCE or TRANSMITTED_LIGHT mode histogram_normalization_center and histogram_normalization_scale must be not None"
            if isinstance(histogram_normalization_center, (list, tuple, np.ndarray)):
                assert len(histogram_normalization_center)==2 and histogram_normalization_center[0]<=histogram_normalization_center[1], "if histogram_normalization_center is a list/tuple it represent a range and should be of length 2"
            if isinstance(histogram_normalization_scale, (list, tuple, np.ndarray)):
                assert len(histogram_normalization_scale)==2 and histogram_normalization_scale[0]<=histogram_normalization_scale[1], "if histogram_normalization_scale is a list/tuple it represent a range and should be of length 2"
        elif histogram_scaling_mode == "PHASE_CONTRAST":
            assert min_histogram_range>0 and min_histogram_range<1, "invalid min_histogram_range"
        elif histogram_scaling_mode == "RANDOM_MIN_MAX" :
            assert min_centile_range is not None, "invalid min range"
            assert max_centile_range is not None, "invalid max range"
            if isinstance(min_centile_range, float):
                min_centile_range = [min_centile_range, min_centile_range]
            if isinstance(max_centile_range, float):
                max_centile_range = [max_centile_range, max_centile_range]
            assert min_centile_range[0]<=min_centile_range[1], "invalid min range"
            assert max_centile_range[0]<=max_centile_range[1], "invalid max range"
            assert min_centile_range[1]<max_centile_range[0], "invalid min and max range"
            self.min_centile_range=min_centile_range
            self.max_centile_range=max_centile_range

        self.histogram_scaling_mode=histogram_scaling_mode
        if gaussian_blur_range is None:
            gaussian_blur_range=0
        self.rotate90=rotate90
        self.min_histogram_range=min_histogram_range
        self.min_histogram_to_zero=min_histogram_to_zero
        self.noise_intensity=noise_intensity
        if np.isscalar(gaussian_blur_range):
            self.gaussian_blur_range=[gaussian_blur_range, gaussian_blur_range]
        else:
            self.gaussian_blur_range=gaussian_blur_range
        self.histogram_voodoo_n_points=histogram_voodoo_n_points
        self.histogram_voodoo_intensity=histogram_voodoo_intensity
        self.illumination_voodoo_n_points=illumination_voodoo_n_points
        self.illumination_voodoo_intensity=illumination_voodoo_intensity
        self.perform_illumination_augmentation = perform_illumination_augmentation
        self.histogram_normalization_center=histogram_normalization_center
        self.histogram_normalization_scale=histogram_normalization_scale
        self.invert=invert
        super().__init__(interpolation_order=interpolation_order, **kwargs)

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)

        if self.rotate90 and img_shape[0]==img_shape[1] and not getrandbits(1):
            params["rotate90"] = True
        # illumination parameters
        if self.perform_illumination_augmentation:
            if self.histogram_scaling_mode=="AUTO" and self.histogram_normalization_center is not None and self.histogram_normalization_scale is not None or self.histogram_scaling_mode=="FLUORESCENCE" or self.histogram_scaling_mode=="TRANSMITTED_LIGHT": # center / scale mode
                if isinstance(self.histogram_normalization_center, (list, tuple, np.ndarray)):
                    params["center"] = uniform(self.histogram_normalization_center[0], self.histogram_normalization_center[1])
                else:
                    params["center"] = self.histogram_normalization_center
                if isinstance(self.histogram_normalization_scale, (list, tuple, np.ndarray)):
                    params["scale"] = uniform(self.histogram_normalization_scale[0], self.histogram_normalization_scale[1])
                else:
                    params["scale"] = self.histogram_normalization_scale
            elif self.histogram_scaling_mode=="RANDOM_MIN_MAX":
                pass
            else: # min max mode
                if self.min_histogram_range<1 and self.min_histogram_range>0:
                    if self.min_histogram_to_zero:
                        params["vmin"] = 0
                        params["vmax"] = uniform(self.min_histogram_range, 1)
                    else:
                        vmin, vmax = pp.compute_histogram_range(self.min_histogram_range)
                        params["vmin"] = vmin
                        params["vmax"] = vmax
                elif self.min_histogram_range==1:
                    params["vmin"] = 0
                    params["vmax"] = 1
            if self.noise_intensity>0:
                poisson, speckle, gaussian = pp.get_random_noise_parameters(self.noise_intensity)
                params["poisson_noise"] = poisson
                params["speckle_noise"] = speckle
                params["gaussian_noise"] = gaussian
            if self.gaussian_blur_range[1]>0 and not getrandbits(1):
                params["gaussian_blur"] = uniform(self.gaussian_blur_range[0], self.gaussian_blur_range[1])

            if self.histogram_voodoo_n_points>0 and self.histogram_voodoo_intensity>0 and not getrandbits(1):
                # draw control points
                if "vmin" in params and "vmax" in params:
                    vmin = params["vmin"]
                    vmax = params["vmax"]
                    control_points = np.linspace(vmin, vmax, num=self.histogram_voodoo_n_points + 2)
                    target_points = pp.get_histogram_voodoo_target_points(control_points, self.histogram_voodoo_intensity)
                    params["histogram_voodoo_target_points"] = target_points
                elif "histogram_voodoo_target_points" in params:
                    del params["histogram_voodoo_target_points"]
            elif "histogram_voodoo_target_points" in params:
                del params["histogram_voodoo_target_points"]
            if self.illumination_voodoo_n_points>0 and self.illumination_voodoo_intensity>0 and not getrandbits(1):
                params["illumination_voodoo_target_points"] = pp.get_illumination_voodoo_target_points(self.illumination_voodoo_n_points, self.illumination_voodoo_intensity)
            elif "illumination_voodoo_target_points" in params:
                del params["illumination_voodoo_target_points"]
        return params

    def apply_transform(self, img, params):
        # geom augmentation
        img = super().apply_transform(img, params)
        if params.get("rotate90", False):
            img = np.rot90(img, k=1, axes=(0, 1))
        # illumination augmentation
        if self.perform_illumination_augmentation:
            img = self._perform_illumination_augmentation(img, params)
        return img

    def _perform_illumination_augmentation(self, img, params):
        if self.histogram_scaling_mode=="TRANSMITTED_LIGHT":
            mean = np.mean(img)
            sd = np.std(img)
            img = (img - (params["center"]+mean)) / (params["scale"] * sd)
        else:
            if "center" in params and "scale" in params:
                img = (img - params["center"]) / params["scale"]
            elif "vmin" in params and "vmax" in params: # PHASE_CONTRAST mode
                min = img.min()
                max = img.max()
                if min==max:
                    raise ValueError("Image is blank, cannot perform illumination augmentation")
                img = pp.adjust_histogram_range(img, min=params["vmin"], max = params["vmax"], initial_range=[min, max])
                if self.invert:
                    img = params["vmin"] + params["vmax"] - img
            else:
                min0, min1, max0, max1 = np.percentile(img, self.min_centile_range+self.max_centile_range)
                cmin = uniform(min0, min1)
                cmax = uniform(max0, max1)
                img = pp.adjust_histogram_range(img, min = 0, max = 1, initial_range=[cmin, cmax]) # will saturate values under cmin or over cmax, as in real life.
                #gmin, min0, min1, max0, max1, gmax = np.percentile(img, [0] + self.min_centile_range+self.max_centile_range + [100])
                # vmin = (gmin - cmin) / (cmax - cmin)
                # vmax = 1 + (gmax - cmax) / (cmax - cmin)
                # img = pp.adjust_histogram_range(img, min = vmin, max = vmax, initial_range=[gmin, gmax])
                if self.invert:
                    img = 1 - img

        if "histogram_voodoo_target_points" in params:
            img = pp.histogram_voodoo(img, self.histogram_voodoo_n_points, self.histogram_voodoo_intensity, target_points = params["histogram_voodoo_target_points"])
        if "illumination_voodoo_target_points" in params:
            target_points = params["illumination_voodoo_target_points"]
            img = pp.illumination_voodoo(img, len(target_points), target_points=target_points)
        if params.get("gaussian_blur", 0)>0:
            img = pp.gaussian_blur(img, params["gaussian_blur"])
        if params.get("poisson_noise", 0)>0:
            img = pp.add_poisson_noise(img, params["poisson_noise"])
        if params.get("speckle_noise", 0)>0:
            img = pp.add_speckle_noise(img, params["speckle_noise"])
        if params.get("gaussian_noise", 0)>0:
            img = pp.add_gaussian_noise(img, params["gaussian_noise"])
        return img
