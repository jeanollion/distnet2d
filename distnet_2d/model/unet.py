
import tensorflow as tf
from .layers import ConvNormAct, Bneck, UpSamplingLayer2D, StopGradient, Combine
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from ..utils.helpers import ensure_multiplicity
from .utils import get_layer_dtype

from .distnet_2d import encoder_op, decoder_op

ENCODER_SETTINGS = [
    [ # l1 = 128 -> 64
        {"filters":32},
        {"filters":64, "downscale":2}
    ],
    [  # l2 64 -> 32
        {"filters":64},
        {"filters":128, "downscale":2}
    ],
    [ # l3: 32->16
        {"filters":128, "kernel_size":5},
        {"filters":128 },
        {"filters":256, "downscale":2},
    ],
    [ # l3: 32->16
        {"filters":256, "kernel_size":5},
        {"filters":256 },
        {"filters":512, "downscale":2},
    ],
]
FEATURE_SETTINGS = [
    {"filters":512},
    {"filters":512},
    {"filters":512},
]

DECODER_SETTINGS = [
    #f, s
    {"filters":32},
    {"filters":64},
    {"filters":128},
    {"filters":256, "up_kernel_size":4}
]


def get_unet(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            skip_combine_mode:str = "conv", # conv, sum
            first_skip_mode:str = None, # sg, omit, None
            skip_stop_gradient:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = DECODER_SETTINGS,
            n_output: int = 1,
            n_output_channels:int = 1,
            n_output_conv_filters:int=32, # 0 = no conv
            name: str="UNet"
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        n_output_channels = ensure_multiplicity(n_output, n_output_channels)
        n_output_conv_filters = ensure_multiplicity(n_output, n_output_conv_filters)
        spatial_dims = input_shape[:-1]

        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, _ = encoder_op(param_list, downsampling_mode=downsampling_mode, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)

        # define feature operations
        feature_convs, _, _, _ = parse_param_list(feature_settings, "FeatureSequence")

        # define decoder operations
        decoder_layers = [decoder_op(**parameters, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode=skip_combine_mode, skip_mode=first_skip_mode if l_idx==0 else ("sg" if skip_stop_gradient else None), activation="relu", layer_idx=l_idx) for l_idx, parameters in enumerate(decoder_settings)]

        # defin output operations
        conv_output = [Conv2D(filters=n_output_conv_filters[i], kernel_size=3, padding='same', activation="relu", name=f"ConvOutput_{i}") if n_output_conv_filters[i]>0 else None for i in range(n_output)]
        output = [Conv2D(filters=n_output_channels[i], kernel_size=3, padding='same', activation="relu", name=f"Output_{i}") for i in range(n_output)]

        # Create GRAPH
        input = tf.keras.layers.Input(shape=input_shape, name="Input")
        residuals = []
        downsampled = [input]
        for l in encoder_layers:
            down, res = l(downsampled[-1])
            downsampled.append(down)
            residuals.append(res)

        feature = downsampled[-1]
        for op in feature_convs:
            feature = op(feature)

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        outputs = [o(conv_o(upsampled[-1])) if conv_o is not None else o(upsampled[-1]) for conv_o, o in zip(conv_output, output)] # TODO add option with 1 decoder per output

        return Model([input], outputs, name=name)
