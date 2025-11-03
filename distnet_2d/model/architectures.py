import math
import copy

from ..utils.helpers import ensure_multiplicity

def get_architecture(architecture_type:str, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    if architecture_type.lower()=="blend":
        n_downsampling = kwargs.pop("n_downsampling", 2)
        if n_downsampling == 2:
            arch = BlendD2
        elif n_downsampling == 3:
            arch = BlendD3
        elif n_downsampling == 4:
            arch = BlendD4
        else:
            raise ValueError(f"Unsupported downsampling number: {n_downsampling}: must be in [2, 3, 4]")
        return arch(**kwargs)
    elif architecture_type.lower()=="TEMA".lower():
        n_downsampling = kwargs.pop("n_downsampling", 3)
        if n_downsampling == 2:
            arch = TemAD2
        elif n_downsampling == 3:
            arch = TemAD3
        elif n_downsampling == 4:
            arch = TemAD4
        else:
            raise ValueError(f"Unsupported downsampling number: {n_downsampling}: must be in [2, 3, 4]")
        return arch(**kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


class ArchBase:
    def __init__(self, filters:int,
                 n_inputs:int=1,
                 spatial_dimensions=None,
                 frame_window:int = 3,
                 category_number: int = 0,  # category for each cell instance (segmentation level), <=1 means do not predict category
                 inference_gap_number: int = 0,
                 tracking:bool = True,  # if false: outputs are only EDM and CDM
                 long_term: bool = True,
                 next: bool = True,
                 early_downsampling:bool = True,
                 self_attention: int = 0,
                 scale_edm:bool = False,
                 batch_norm:bool = True, dropout:float=0.2, l2_reg:float=0,
                 downsampling_mode="maxpool_and_stride", upsampling_mode ="tconv", skip_combine_mode:str="conv",  #conv, wsconv
                 attention_filters:int = 0, attention:int = 0, attention_positional_encoding:str="2d",
                 combine_kernel_size:int=1, pair_combine_kernel_size:int=5,
                 skip_connections=[-1], skip_stop_gradient:bool = False,
                 frame_aware:bool=False, frame_max_distance:int=0,
                 predict_fw: bool = True, predict_edm_derivatives:bool = False, predict_cdm_derivatives:bool = False,
                 ):
        if attention > 0 or self_attention:
            assert spatial_dimensions is not None and min(spatial_dimensions) > 0, f"for attention mechanism, spatial dim must be provided. Got {spatial_dimensions}"
        self.spatial_dimensions=spatial_dimensions
        self.n_inputs = n_inputs
        self.frame_window = frame_window
        self.long_term=long_term
        self.category_number=category_number
        self.tracking=tracking
        self.inference_gap_number=inference_gap_number
        self.future_frames=next
        self.scale_edm=scale_edm
        self.skip_connections = skip_connections
        self.skip_stop_gradient=skip_stop_gradient
        self.attention = attention
        self.attention_filters = attention_filters
        self.attention_positional_encoding = attention_positional_encoding
        self.combine_kernel_size = combine_kernel_size
        self.pair_combine_kernel_size = pair_combine_kernel_size
        self.downsampling_mode = downsampling_mode
        self.upsampling_mode = upsampling_mode
        self.skip_combine_mode=skip_combine_mode
        self.frame_aware=frame_aware
        self.frame_max_distance = frame_max_distance
        self.filters = filters
        self.early_downsampling = early_downsampling
        self.self_attention = self_attention
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.l2_reg=l2_reg
        self.predict_fw=predict_fw
        self.predict_edm_derivatives=predict_edm_derivatives
        self.predict_cdm_derivatives=predict_cdm_derivatives
        # to be defined
        self.encoder_settings = None
        self.feature_settings = None
        self.decoder_settings = None
        self.feature_decoder_settings = None


class ArchDepth(ArchBase):
    def __init__(self, filters:int, **kwargs):
        super().__init__(filters, **kwargs)


class D2(ArchDepth):
    def __init__(self, kernel_size_fd:int=5, **kwargs):
        super().__init__(**kwargs)
        if self.attention>0 or self.self_attention>0:
            print(f"spatial dimension at attention layer: {self.spatial_dimensions[0] / 2**2} x {self.spatial_dimensions[1] / 2**2}")
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2)
        ker1_2, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, 2)
        ker2, dil2 = get_kernels_and_dilation(5, 2, self.spatial_dimensions, 2 * 2)
        ker2_2, dil2_2 = get_kernels_and_dilation(5, 3, self.spatial_dimensions, 2 * 2)
        ker2_3, dil2_3 = get_kernels_and_dilation(5, 4, self.spatial_dimensions, 2 * 2)
        ker2_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, 2 * 2)
        self.encoder_settings = [
            [
                {"filters": 32, "op": "conv", "kernel_size": ker0, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "kernel_size": ker0, "downscale": 2, "weight_scaled": False, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "op": "conv", "kernel_size": ker1, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "op": "conv", "kernel_size": ker1_2, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": self.filters, "kernel_size": ker1, "downscale": 2, "weight_scaled": False, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)
        self.feature_settings = [
            {"op": "res2d", "dilation": dil2, "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil2 if self.self_attention > 0 else dil2_2,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "res2d", "attention_filters": self.attention_filters,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_3,
             "dilation": dil2 if self.self_attention > 0 else dil2_3, "dropout_rate": self.dropout,
             "num_attention_heads": self.self_attention},
            {"op": "res2d", "dilation": dil2, "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil2 if self.self_attention > 0 else dil2_2,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size": ker2_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "res2d", "kernel_size": ker2_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker2_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": 0,
             "batch_norm": self.batch_norm}
        ]
        self.decoder_settings = [
            {"filters": 16, "op": "conv", "n_conv": 0, "conv_kernel_size": 4, "up_kernel_size": 4,
             "weight_scaled_up": False, "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 32, "op": "res2d", "conv_kernel_size":ker1, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0}
        ]


class D3(ArchDepth):
    def __init__(self, kernel_size_fd:int=5, **kwargs):
        super().__init__(**kwargs)
        if self.attention>0 or self.self_attention>0:
            print(f"spatial dimension at attention layer: {self.spatial_dimensions[0] / 2**3} x {self.spatial_dimensions[1] / 2**3}")
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2)
        ker2, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2 * 2)
        ker3, dil3 = get_kernels_and_dilation(5, 2, self.spatial_dimensions, 2 ** 3)
        ker3_2, dil3_2 = get_kernels_and_dilation(5, 3, self.spatial_dimensions, 2 ** 3)
        ker3_3, dil3_3 = get_kernels_and_dilation(5, 4, self.spatial_dimensions, 2 ** 3)
        ker3_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, 2 ** 3)
        self.encoder_settings = [
            [
                {"filters": 32, "op": "conv", "kernel_size": ker0, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "kernel_size": ker0, "downscale": 2, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "kernel_size": ker1, "dropout_rate": 0},
                {"filters": 64, "kernel_size": ker1, "downscale": 2, "dropout_rate": 0}
            ],
            [
                {"filters": 64, "op": "res2d", "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0},
                {"filters": 64, "op": "res2d", "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0},
                {"filters": self.filters, "kernel_size": ker2, "downscale": 2, "weight_scaled": False, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)

        self.feature_settings = [
            {"op": "res2d", "dilation": dil3, "kernel_size": ker3, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil3 if self.self_attention > 0 else dil3_2,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "res2d", "attention_filters": self.attention_filters,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_3,
             "dilation": dil3 if self.self_attention > 0 else dil3_3, "dropout_rate": self.dropout},
            {"op": "res2d", "dilation": dil3, "kernel_size": ker3, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil3 if self.self_attention > 0 else dil3_2,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker3, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size": ker3_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "res2d", "kernel_size": ker3_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker3_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": 0,
             "batch_norm": self.batch_norm}
        ]
        self.decoder_settings = [
            {"filters": 16, "op": "conv", "n_conv": 0, "conv_kernel_size": 4, "up_kernel_size": 4,
             "weight_scaled_up": False, "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 32, "op": "res2d", "conv_kernel_size" : ker1, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0},
            {"filters": 64, "op": "res2d", "conv_kernel_size" : ker2, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0}
        ]


class D4(ArchDepth):
    def __init__(self, kernel_size_fd:int=5, **kwargs):
        super().__init__(**kwargs)
        if self.attention>0 or self.self_attention>0:
            print(f"spatial dimension at attention layer: {self.spatial_dimensions[0] / 2**4} x {self.spatial_dimensions[1] / 2**4}")
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2)
        ker2, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, 2 ** 2)
        ker2_1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2 ** 2)
        ker3, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, 2 ** 3)
        ker3_2, dil3_2 = get_kernels_and_dilation(5, 2, self.spatial_dimensions, 2 ** 3)
        ker3_3, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 2 ** 3)
        ker4, dil4 = get_kernels_and_dilation(5, 2, self.spatial_dimensions, 2 ** 4)
        ker4_2, dil4_2 = get_kernels_and_dilation(5, 3, self.spatial_dimensions, 2 ** 4)
        ker4_3, dil4_3 = get_kernels_and_dilation(5, 4, self.spatial_dimensions, 2 ** 4)
        ker4_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, 2 ** 4)
        self.encoder_settings = [
            [
                {"filters": 16, "op": "conv", "kernel_size": ker0, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 16, "kernel_size": ker0, "downscale": 2, "dropout_rate": 0}
            ],
            [
                {"filters": 16, "kernel_size": ker1, "dropout_rate": 0},
                {"filters": 32, "kernel_size": ker1, "downscale": 2, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "op": "res2d", "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0},
                {"filters": 32, "op": "res2d", "kernel_size": ker2, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0},
                {"filters": 64, "kernel_size": ker2, "downscale": 2, "weight_scaled": False, "dropout_rate": 0,
                 "batch_norm": False}
            ],
            [
                {"filters": 64, "op": "res2d", "kernel_size": ker3, "weighted_sum": False, "weight_scaled": False,
                 "dropout_rate": 0},
                {"filters": 64, "op": "res2d", "kernel_size": ker3_2, "dilation": dil3_2, "weighted_sum": False,
                 "weight_scaled": False, "dropout_rate": 0},
                {"filters": self.filters, "kernel_size": ker3, "downscale": 2, "weight_scaled": False, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)

        self.feature_settings = [
            {"op": "res2d", "dilation": dil4, "kernel_size": ker4, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil4 if self.self_attention > 0 else dil4_2,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "res2d", "attention_filters": self.attention_filters,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_3,
             "dilation": dil4 if self.self_attention > 0 else dil4_3, "dropout_rate": self.dropout},
            {"op": "res2d", "dilation": dil4, "kernel_size": ker4, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "res2d", "dilation": dil4 if self.self_attention > 0 else dil4_2,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_2, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker4, "weighted_sum": False, "weight_scaled": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size":ker4_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "res2d", "kernel_size":ker4_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size":ker4_fd, "weighted_sum": False, "weight_scaled": False, "dropout_rate": 0,
             "batch_norm": self.batch_norm}
        ]
        self.decoder_settings = [
            {"filters": 16, "op": "conv", "n_conv": 0, "conv_kernel_size": 4, "up_kernel_size": 4,
             "weight_scaled_up": False, "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 16, "op": "res2d", "conv_kernel_size": ker1, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0},
            {"filters": 32, "op": "res2d", "conv_kernel_size": ker2_1, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0},
            {"filters": 64, "op": "res2d", "conv_kernel_size": ker3_3, "weighted_sum": False, "n_conv": 2, "up_kernel_size": 4,
             "weight_scaled_up": False, "weight_scaled": False, "batch_norm": False, "dropout_rate": 0}
        ]


class Blend(ArchBase):
    def __init__(self, frame_aware:bool, blending_filter_factor:float=0.5, **kwargs):
        super().__init__(frame_aware=frame_aware, **kwargs)
        self.blending_filter_factor = blending_filter_factor
        # to be defined
        self.feature_blending_settings = None


class BlendD2(Blend, D2):
    def __init__(self, filters:int = 128, **kwargs):
        super().__init__(filters=filters, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD2-{filters}"
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False, "split_conv":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False, "split_conv":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False, "split_conv":False}
        ]


class BlendD3(Blend, D3):
    def __init__(self, filters:int = 192, **kwargs):
        super().__init__(filters=filters, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD3-{filters}"
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False}
        ]


class BlendD4(Blend, D4):
    def __init__(self, filters:int = 192, **kwargs):
        super().__init__(filters=filters, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD3-{filters}"
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":self.dropout, "batch_norm":False}
        ]


class TemA(ArchBase):
    def __init__(self, **kwargs):
        super().__init__(frame_aware=True, pair_combine_kernel_size=1, **kwargs)
        assert self.attention > 0, "attention heads cannot be null"


class TemAD2(TemA, D2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TemAD3(TemA, D3):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class TemAD4(TemA, D4):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_kernels_and_dilation(target_kernel, target_dilation, spa_dimensions, downsampling):
    if spa_dimensions is None:
        return target_kernel, target_dilation
    spa_dimensions = ensure_multiplicity(2, spa_dimensions)
    kernel = ensure_multiplicity(2, target_kernel)
    dilation = ensure_multiplicity(2, target_dilation)
    spa_dimensions = [d/downsampling if d is not None and d>0 else None for d in spa_dimensions]
    for i in range(len(spa_dimensions)):
        while not test_ker_dil(kernel[i], dilation[i], spa_dimensions[i]):
            if dilation[i] > 1:
                dilation[i] -=1
            elif kernel[i] > 1:
                kernel[i] = 1 + 2 * ((kernel[i] - 1) // 2 - 1)
            else:
                raise ValueError(f"Cannot find kernel size that suit dimension: {spa_dimensions[i]}")
    kernel = kernel[0] if kernel[0] == kernel[1] else kernel
    dilation = dilation[0] if dilation[0] == dilation[1] else dilation
    return kernel, dilation


def test_ker_dil(ker, dil, dim):
    if ker == 1 and dil == 1 or dim is None or dim <= 0:
        return True
    size = (ker-1)*dil
    return dim >= size * 2