from keras_preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator # this version doesn't have interpolation_order
import numpy as np
from math import tan, atan, pi, copysign
import distnet_2d.data.pre_processing as pp
from random import getrandbits, uniform, choice
import copy
import scipy.ndimage as ndi

class ImageDataGenerator2D(ImageDataGenerator):
    def __init__(self, rotate90=False, perform_illumination_augmentation = True, gaussian_blur_range=[1, 2], noise_intensity = 0.1, min_histogram_range=0.1, min_histogram_to_zero=False, histogram_normalization_center=None, histogram_normalization_scale=None, histogram_voodoo_n_points=5, histogram_voodoo_intensity=0.5, illumination_voodoo_n_points=5, illumination_voodoo_intensity=0.6, **kwargs):
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
        super().__init__(**kwargs)

    def get_random_transform(self, img_shape, seed=None):
        params = super().get_random_transform(img_shape, seed)

        if self.rotate90 and img_shape[0]==img_shape[1] and not getrandbits(1):
            params["rotate90"] = True
        # illumination parameters
        if self.perform_illumination_augmentation:
            if self.histogram_normalization_center is not None and self.histogram_normalization_scale is not None: # center / scale mode
                if isinstance(self.histogram_normalization_center, (list, tuple, np.ndarray)):
                    assert len(self.histogram_normalization_center)==2, "if histogram_normalization_center is a list/tuple it represent a range and should be of length 2"
                    params["center"] = uniform(self.histogram_normalization_center[0], self.histogram_normalization_center[1])
                else:
                    params["center"] = self.histogram_normalization_center
                if isinstance(self.histogram_normalization_scale, (list, tuple, np.ndarray)):
                    assert len(self.histogram_normalization_scale)==2, "if histogram_normalization_scale is a list/tuple it represent a range and should be of length 2"
                    params["scale"] = uniform(self.histogram_normalization_scale[0], self.histogram_normalization_scale[1])
                else:
                    params["scale"] = self.histogram_normalization_scale
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
        img = self._perform_illumination_augmentation(img, params)
        return img

    def _perform_illumination_augmentation(self, img, params):
        if "center" in params and "scale" in params:
            img = (img - params["center"]) / params["scale"]
        elif "vmin" in params and "vmax" in params:
            min = img.min()
            max = img.max()
            if min==max:
                raise ValueError("Image is blank, cannot perform illumination augmentation")
            img = pp.adjust_histogram_range(img, min=params["vmin"], max = params["vmax"], initial_range=[min, max])
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
