def get_architecture(architecture_type:str, **kwargs):
    if architecture_type.lower()=="blend":
        arch = BlendD2 if kwargs.pop("n_downsampling", 2) == 2 else BlendD3
        return arch(**kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")

class BlendD2():
    def __init__(self, filters:int = 128, blending_filter_factor:float=0.5, batch_norm:bool = True, dropout:float=0.2, self_attention:bool = False, attention:bool = False, combine_kernel_size:int=1, pair_combine_kernel_size:int=5, skip_connections=[-1]):
        prefix = f"{'a' if attention else ''}{'sa' if self_attention else ''}"
        self.name = f"{prefix}blendD2-{filters}"
        self.skip_connections=skip_connections
        self.attention = attention
        self.combine_kernel_size = combine_kernel_size
        self.pair_combine_kernel_size = pair_combine_kernel_size
        self.blending_filter_factor=blending_filter_factor
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":32, "op":"conv", "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0, "batch_norm":False}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":2 if attention else 3, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":filters, "op":"selfattention" if attention else "res2d", "kernel_size":5, "dilation":2 if attention else 4, "dropout_rate":0 if attention else dropout },
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":2 if attention else 3, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm},
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
    def __init__(self, filters:int = 192, blending_filter_factor:float=0.5, batch_norm:bool = True, dropout:float=0.2, self_attention:bool = False, attention:bool = False, combine_kernel_size:int=1, pair_combine_kernel_size:int=5, skip_connections=[-1]):
        prefix = f"{'a' if attention else ''}{'sa' if self_attention else ''}"
        self.name = f"{prefix}blendD3-{filters}"
        self.skip_connections=skip_connections
        self.attention = attention
        self.combine_kernel_size = combine_kernel_size
        self.pair_combine_kernel_size = pair_combine_kernel_size
        self.blending_filter_factor = blending_filter_factor
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "dropout_rate":0}
            ],
            [
                {"filters":32, "dropout_rate":0},
                {"filters":64, "downscale":2, "dropout_rate":0}
            ],
            [
                {"filters":64, "op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0, "batch_norm":False}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":2 if self_attention else 3, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":filters, "op":"selfattention" if self_attention else "res2d", "kernel_size":5, "dilation":2 if self_attention else 4, "dropout_rate":0 if self_attention else dropout },
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "dilation":2 if self_attention else 3, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm},
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
