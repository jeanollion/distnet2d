import math
import copy

from ..utils.helpers import ensure_multiplicity
def get_architecture(architecture_type:str, **kwargs):
    kwargs = copy.deepcopy(kwargs)
    if architecture_type.lower()=="blend":
        arch = BlendD2 if kwargs.pop("n_downsampling", 2) == 2 else BlendD3
        return arch(**kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

class BlendD2():
    def __init__(self, filters:int = 128, blending_filter_factor:float=0.5, batch_norm:bool = True, dropout:float=0.2, self_attention:int = 0, attention:int = 0, combine_kernel_size:int=1, pair_combine_kernel_size:int=5, skip_connections=[-1], spatial_dimensions=None):
        prefix = f"{'a' if attention else ''}{'sa' if self_attention else ''}"
        self.name = f"{prefix}blendD2-{filters}"
        self.skip_connections=skip_connections
        self.attention = attention
        self.self_attention=self_attention
        self.dropout=dropout
        self.combine_kernel_size = combine_kernel_size
        self.pair_combine_kernel_size = pair_combine_kernel_size
        self.blending_filter_factor=blending_filter_factor
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        ker1, _ = get_kernels_and_dilation(3, 1, spatial_dimensions, 2 )
        ker1_2, _ = get_kernels_and_dilation(5, 1, spatial_dimensions, 2)
        ker2, dil2 = get_kernels_and_dilation(5, 2, spatial_dimensions, 2 * 2)
        ker2_2, dil2_2 = get_kernels_and_dilation(5, 3, spatial_dimensions, 2 * 2)
        ker2_3, dil2_3 = get_kernels_and_dilation(5, 4, spatial_dimensions, 2 * 2)
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "op":"conv", "kernel_size":ker1, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":32, "op":"conv", "kernel_size":ker1_2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":filters, "kernel_size":ker1, "downscale":2, "weight_scaled":False, "dropout_rate":0, "batch_norm":False}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":dil2, "kernel_size":ker2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":dil2 if self_attention>0 else dil2_2, "kernel_size":ker2 if self_attention>0 else ker2_2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":filters, "op":"selfattention" if self_attention>0 else "res2d", "kernel_size":ker2 if self_attention>0 else ker2_3, "dilation":dil2 if self_attention>0 else dil2_3, "dropout_rate":dropout, "num_attention_heads":self_attention },
            {"op":"res2d", "dilation":dil2, "kernel_size":ker2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":dil2 if self_attention>0 else dil2_2, "kernel_size":ker2 if self_attention>0 else ker2_2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "kernel_size":ker2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False, "split_conv":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False, "split_conv":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False, "split_conv":False}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class BlendD3():
    def __init__(self, filters:int = 192, blending_filter_factor:float=0.5, batch_norm:bool = True, dropout:float=0.2, self_attention:int = 0, attention:int = 0, combine_kernel_size:int=1, pair_combine_kernel_size:int=5, skip_connections=[-1], spatial_dimensions=None):
        prefix = f"{'a' if attention else ''}{'sa' if self_attention else ''}"
        self.name = f"{prefix}blendD3-{filters}"
        self.skip_connections=skip_connections
        self.attention = attention
        self.self_attention=self_attention
        self.dropout=dropout
        self.combine_kernel_size = combine_kernel_size
        self.pair_combine_kernel_size = pair_combine_kernel_size
        self.blending_filter_factor = blending_filter_factor
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        ker2, _ = get_kernels_and_dilation(3, 1, spatial_dimensions, 2 * 2)
        ker3, dil3 = get_kernels_and_dilation(5, 2, spatial_dimensions, 2 * 2 * 2)
        ker3_2, dil3_2 = get_kernels_and_dilation(5, 3, spatial_dimensions, 2 * 2 * 2)
        ker3_3, dil3_3 = get_kernels_and_dilation(5, 4, spatial_dimensions, 2 * 2 * 2)
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "dropout_rate":0}
            ],
            [
                {"filters":32, "dropout_rate":0},
                {"filters":64, "downscale":2, "dropout_rate":0}
            ],
            [
                {"filters":64, "op":"res2d", "kernel_size":ker2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "kernel_size":ker2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":filters, "kernel_size":ker2, "downscale":2, "weight_scaled":False, "dropout_rate":0, "batch_norm":False}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":dil3, "kernel_size":ker3, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":dil3 if self_attention>0 else dil3_2, "kernel_size":ker3 if self_attention>0 else ker3_2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":filters, "op":"selfattention" if self_attention>0 else "res2d", "kernel_size":ker3 if self_attention>0 else ker3_3, "dilation":dil3 if self_attention>0 else dil3_3, "dropout_rate":dropout },
            {"op":"res2d", "dilation":dil3, "kernel_size":ker3, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":dil3 if self_attention>0 else dil3_2, "kernel_size":ker3 if self_attention>0 else ker3_2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "kernel_size":ker3, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0},
            {"filters":64, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

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