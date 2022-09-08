import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion, distance_transform_edt
from scipy.ndimage import find_objects
import random
from random import uniform, random, randint, getrandbits
from scipy import interpolate
import copy
from scipy.ndimage.filters import generic_filter
try:
	import edt
except Exception:
	pass
try:
    import dataset_iterator.helpers as dih
except:
    dih=None
from ..utils.helpers import ensure_multiplicity

def batch_wise_fun(fun):
	#return lambda batch : np.stack([fun(batch[i]) for i in range(batch.shape[0])], 0)
    def func(batch):
        for b in range(batch.shape[0]):
            batch[b] = fun(batch[b])
        return batch
    return func

def apply_and_stack_channel(*funcs):
	return lambda batch : np.concatenate([fun(batch) for fun in funcs], -1)

def identity(batch):
	return batch

def weight_map_mask_class_balance(batch, max_background_ratio=0, set_background_to_one=False, dtype=np.float32):
	wm = np.ones(shape = batch.shape, dtype=dtype)
	if max_background_ratio<0:
		return wm
	n_nonzeros = np.count_nonzero(batch)
	if n_nonzeros!=0:
		n_tot = np.prod(batch.shape)
		p_back = (n_tot - n_nonzeros) / n_tot
		background_ratio = (n_tot - n_nonzeros) / n_nonzeros
		if max_background_ratio>0 and background_ratio>max_background_ratio:
			p_back = max_background_ratio / (1 + max_background_ratio)
		if set_background_to_one:
			wm[batch!=0] = p_back / (1 - p_back)
		else:
			wm[batch!=0] = p_back
			wm[batch==0] = 1-p_back
	return wm

def binarize(img, dtype=np.float32):
	return np.where(img, dtype(1), dtype(0))

def sometimes(func, prob=0.5):
    return lambda im:func(im) if random()<prob else im

def apply_successively(*functions):
    if len(functions)==0:
        return lambda img:img
    def func(img):
        for f in functions:
            img = f(img)
        return img
    return func

def gaussian_blur(img, sig):
    if len(img.shape)>2 and img.shape[-1]==1:
        return np.expand_dims(gaussian_filter(img.squeeze(-1), sig), -1)
    else:
        return gaussian_filter(img, sig)

def random_gaussian_blur(img, sig_min=1, sig_max=2):
    sig = uniform(sig_min, sig_max)
    return gaussian_blur(img, sig)

def adjust_histogram_range(img, min=0, max=1, initial_range=None):
    if initial_range is None:
        initial_range=[img.min(), img.max()]
    return np.interp(img, initial_range, (min, max))

def compute_histogram_range(min_range, range=[0, 1]):
    if range[1]-range[0]<min_range:
        raise ValueError("Range must be superior to min_range")
    vmin = uniform(range[0], range[1]-min_range)
    vmax = uniform(vmin+min_range, range[1])
    return vmin, vmax

def random_histogram_range(img, min_range=0.1, range=[0,1]):
    min, max = compute_histogram_range(min_range, range)
    return adjust_histogram_range(img, min, max)

def random_scaling(img, center=None, scale=None, alpha_range=[-0.3, 0.17], beta_range=0.07):
    """Scales the image by this formlua: I' = ( I - ( μ + ( β * std ) ) ) / (std * 10**α). α, β randomly chosen

    Parameters
    ----------
    img : numpy array
    center : float
        default center value, if center, mean is computed on the array
    scale : float
        default standard deviation value, if none, std is computed on the array
    alpha_range : type
        range in which α is uniformly chosen (if scalar: range is [-alpha_range, alpha_range])
    beta_range : type
        range in which β is uniformly chosen (if scalar: range is [-beta_range, beta_range])

    Returns
    -------
    type
        scaled array

    """
    if center is None:
        center = img.mean()
    if scale is None:
        scale = img.std()
    if np.isscalar(alpha_range):
        alpha_range = [-alpha_range, alpha_range]
    if np.isscalar(beta_range):
        beta_range = [-beta_range, beta_range]
    factor = 1. / (scale * 10**uniform(alpha_range[0], alpha_range[1]))
    center = center + scale * uniform(beta_range[0], beta_range[1])
    return (img - center) * factor

def add_gaussian_noise(img, sigma=[0, 0.1], scale_sigma_to_image_range=True):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    if scale_sigma_to_image_range:
        sigma *= (img.max() - img.min())
    gauss = np.random.normal(0,sigma,img.shape)
    return img + gauss

def add_speckle_noise(img, sigma=[0, 0.1]):
    if is_list(sigma):
        if len(sigma)==2:
            sigma = uniform(sigma[0], sigma[1])
        else:
            raise ValueError("Sigma  should be either a list/tuple of lenth 2 or a scalar")
    min = img.min()
    gauss = np.random.normal(1, sigma, img.shape)
    return (img - min) * gauss + min

def add_poisson_noise(img, noise_intensity=[0, 0.1], adjust_intensity=True):
    if is_list(noise_intensity):
        if len(noise_intensity)==2:
            noise_intensity = uniform(noise_intensity[0], noise_intensity[1])
        else:
            raise ValueError("noise_intensity should be either a list/tuple of lenth 2 or a scalar")
    if adjust_intensity:
        noise_intensity /= 10.0 # so that intensity is comparable to gaussian sigma
    min = img.min()
    max = img.max()
    img = (img - min) / (max - min)
    output = np.random.poisson(img / noise_intensity) * noise_intensity
    return output * (max - min) + min

def noise_function(noise_max_intensity=0.15):
    def res(img):
        gauss = not getrandbits(1)
        speckle = not getrandbits(1)
        poisson = not getrandbits(1)
        ni = noise_max_intensity / float(1.5 ** (sum([gauss, speckle, poisson]) - 1))
        funcs = []
        if poisson:
            funcs.append(lambda im : add_poisson_noise(im, noise_intensity=[0, ni]))
        if speckle:
            funcs.append(lambda im:add_speckle_noise(im, sigma=[0, ni]))
        if gauss:
            funcs.append(lambda im:add_gaussian_noise(im, sigma=[0, ni * 0.7]))
        return apply_successively(*funcs)(img)
    return res

def get_random_noise_parameters(noise_max_intensity=0.15):
    gauss = not getrandbits(1)
    speckle = not getrandbits(1)
    poisson = not getrandbits(1)
    ni = noise_max_intensity / float(1.5 ** (sum([gauss, speckle, poisson]) - 1))

    gauss_i = uniform(0, noise_max_intensity * 0.7) if gauss else 0
    speckle_i = uniform(0, noise_max_intensity) if speckle else 0
    poisson_i = uniform(0, noise_max_intensity) if poisson else 0

    return poisson_i, speckle_i, gauss_i

def grayscale_deformation_function(add_noise=True, adjust_histogram_range=True):
    funcs = [sometimes(random_gaussian_blur), sometimes(histogram_voodoo), sometimes(illumination_voodoo)]
    if add_noise:
        funcs.append(noise_function())
    if adjust_histogram_range:
        funcs.append(lambda img:random_histogram_range(img))
    return lambda img:apply_successively(*funcs)(img)

def is_list(l):
    return isinstance(l, (list, tuple, np.ndarray))

def histogram_voodoo(image, num_control_points=5, intensity=0.5, target_points = None, return_mapping = False):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It performs an elastic deformation on the image histogram to simulate
    changes in illumination
    '''

    if target_points is not None and len(target_points)!=num_control_points+2:
        raise ValueError("invalid target_point number")
    if target_points is None and intensity<=0 or intensity>=1:
        raise ValueError("Intensity should be in range ]0, 1[")

    min = image.min()
    max = image.max()
    control_points = np.linspace(min, max, num=num_control_points + 2)
    if target_points is None:
        target_points = get_histogram_voodoo_target_points(control_points, intensity)
    elif target_points[0] != min or target_points[-1] != max:
        #print("target points borders differs: [{};{}] tp: {}".format(min, max, target_points))
        target_points[0] = min
        target_points[-1] = max
    mapping = interpolate.PchipInterpolator(control_points, target_points)
    newimage = mapping(image)
    if return_mapping:
        return newimage, mapping
    else:
        return newimage

def get_histogram_voodoo_target_points(control_points, intensity):
    if intensity<=0 or intensity>=1:
        raise ValueError("Intensity should be in range ]0, 1[")
    min = control_points[0]
    max = control_points[-1]
    num_control_points = len(control_points) - 2
    delta = intensity * (max - min) / float(num_control_points + 1)
    target_points = copy.copy(control_points)
    target_points += np.random.uniform(low=-delta, high=delta, size = len(target_points))
    return target_points

def illumination_voodoo(image, num_control_points=5, intensity=0.8, target_points = None):
    '''
    Adapted from delta software: https://gitlab.com/dunloplab/delta/blob/master/data.py
    It simulates a variation in illumination along the length of the chamber
    '''
    if intensity>=1 or intensity<=0:
        raise ValueError("Intensity should be in range ]0, 1[")
    if target_points is not None and len(target_points)!=num_control_points:
        raise ValueError("invalid target_point number")
    # Create a random curve along the length of the chamber:
    control_points = np.linspace(0, image.shape[0]-1, num=num_control_points)
    if target_points is None:
        target_points = get_illumination_voodoo_target_points(num_control_points, intensity)
    mapping = interpolate.PchipInterpolator(control_points, target_points)
    curve = mapping(np.linspace(0,image.shape[0]-1,image.shape[0]))
    curveIm = np.reshape( np.tile( np.reshape(curve, curve.shape + (1,)), (1, image.shape[1])) ,image.shape )
    # Apply this curve to the image intensity along the length of the chamebr:
    min = image.min()
    max = image.max()
    newimage = np.multiply(image-min, curveIm)
    # Rescale values to original range:
    newimage = np.interp(newimage, (newimage.min(), newimage.max()), (min, max))

    return newimage

def get_illumination_voodoo_target_points(num_control_points, intensity):
    if intensity>=1 or intensity<=0:
        raise ValueError("Intensity should be in range ]0, 1[")
    return np.random.uniform(low=(1 - intensity) / 2.0, high=(1 + intensity) / 2.0, size=num_control_points)

def get_histogram_normalization_center_scale_ranges(histogram, bins, center_percentile_extent, scale_percentile_range, verbose=False):
    assert dih is not None, "dataset_iterator package is required for this method"
    mode_value = dih.get_modal_value(histogram, bins)
    mode_percentile = dih.get_percentile_from_value(histogram, bins, mode_value)
    print("model value={}, model percentile={}".format(mode_value, mode_percentile))
    assert mode_percentile<scale_percentile_range[0], "mode percentile is {} and must be lower than lower bound of scale_percentile_range={}".format(mode_percentile, scale_percentile_range)
    percentiles = [max(0, mode_percentile-center_percentile_extent), min(100, mode_percentile+center_percentile_extent)]
    scale_percentile_range = ensure_multiplicity(2, scale_percentile_range)
    if isinstance(scale_percentile_range, tuple):
        scale_percentile_range = list(scale_percentile_range)
    percentiles = percentiles + scale_percentile_range
    values = dih.get_percentile(histogram, bins, percentiles)
    mode_range = [values[0], values[1] ]
    scale_range = [values[2] - mode_value, values[3] - mode_value]
    if verbose:
        print("normalization_center_scale: modal value: {}, center_range: [{}; {}] scale_range: [{}; {}]".format(mode_value, mode_range[0], mode_range[1], scale_range[0], scale_range[1]))
    return mode_range, scale_range

def get_center_scale_range(dataset, raw_feature_name:str = "/raw", fluorescence:bool = False, tl_sd_factor:float=3., fluo_centile_range:list=[75, 99.9], fluo_centile_extent:float=5, transmitted_light_per_image_mode:bool=True):
    """Computes a range for center and for scale factor for data augmentation.
    Image can then be normalized using a random center C in the center range and a random scaling factor in the scale range: I -> (I - C) / S

    Parameters
    ----------
    dataset : datasetIO/path(str) OR list/tuple of datasetIO/path(str)
    raw_feature_name : str
        name of the dataset
    fluorescence : bool
        in fluoresence mode:
            mode M is computed, corresponding to the Mp centile: M = centile(Mp). center_range = [centile(Mp-fluo_centile_extent), centile(Mp+fluo_centile_extent)]
            scale_range = [centile(fluo_centile_range[0]) - M, centile(fluo_centile_range[0]) + M ]
        in transmitted light mode: with transmitted_light_per_image_mode=True center_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd]; scale_range = [sd/tl_sd_factor, sd*tl_sd_factor]
        in transmitted light mode: with transmitted_light_per_image_mode=False: center_range = [-tl_sd_factor*sd, tl_sd_factor*sd]; scale_range = [1/tl_sd_factor, tl_sd_factor]
    tl_sd_factor : float
        use in the computation of transmitted light ranges cf description of fluorescence parameter
    fluo_centile_range : list
        in fluoresence mode, interval for scale range in centiles
    fluo_centile_extent : float
        in fluoresence mode, extent for center range in centiles

    Returns
    -------
    scale_range (list(2)) , center_range (list(2))

    """
    if isinstance(dataset, (list, tuple)):
        scale_range, center_range = [], []
        for ds in dataset:
            sr, cr = get_center_scale_range(ds, raw_feature_name, fluorescence, tl_sd_factor, fluo_centile_range, fluo_centile_extent)
            scale_range.append(sr)
            center_range.append(cr)
        if len(dataset)==1:
            return scale_range[0], center_range[0]
        return scale_range, center_range
    if fluorescence:
        bins = dih.get_histogram_bins_IPR(*dih.get_histogram(dataset, raw_feature_name, bins=1000), n_bins=256, percentiles=[0, 95], verbose=True)
        histo, _ = dih.get_histogram(dataset, raw_feature_name, bins=bins)
        center_range, scale_range = get_normalization_center_scale_ranges(histo, bins, fluo_centile_extent, fluo_centile_range, verbose=True)
        print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
    else:
        mean, sd = dih.get_mean_sd(dataset, raw_feature_name, per_channel=True)
        mean, sd = np.mean(mean), np.mean(sd)
        print("mean: {} sd: {}".format(mean, sd))
        if transmitted_light_per_image_mode:
            center_range, scale_range = [- tl_sd_factor*sd, tl_sd_factor*sd], [1./tl_sd_factor, tl_sd_factor]
            print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        else:
            center_range, scale_range = [mean - tl_sd_factor*sd, mean + tl_sd_factor*sd], [sd/tl_sd_factor, sd*tl_sd_factor]
            print("center: [{}; {}] / scale: [{}; {}]".format(center_range[0], center_range[1], scale_range[0], scale_range[1]))
        return center_range, scale_range
