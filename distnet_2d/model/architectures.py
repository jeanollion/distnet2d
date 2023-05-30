class ASABlendD3v2():
    def __init__(self, filters:int = 192, batch_norm:bool = True, dropout:float=0.1):
        self.name = f"asa-blend2-{filters}"
        self.attention = True
        self.combine_kernel_size = 1
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":64, "op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "batch_norm":batch_norm, "dropout_rate":dropout},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "batch_norm":batch_norm, "dropout_rate":dropout},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False,"dropout_rate":0},
            {"filters":64, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class ASABlendD2v2():
    def __init__(self, filters:int = 128, batch_norm:bool = True, dropout:float=0.1):
        self.name = f"asa-blend2-d2-{filters}"
        self.attention = True
        self.combine_kernel_size = 1
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":32, "op":"conv", "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0, "batch_norm":False}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0, "batch_norm":False},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class ASABlendD3v3():
    def __init__(self, filters:int = 192, batch_norm:bool = True, dropout:float=0.1):
        self.name = f"asa-blend2-{filters}"
        self.attention = True
        self.combine_kernel_size = 1
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":64, "op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":64, "op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout},
            {"op":"res2d", "dilation":2, "weighted_sum":False, "batch_norm":False, "dropout_rate":dropout},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout},
            {"filters":1., "op":"conv", "weighted_sum":False, "batch_norm":batch_norm, "dropout_rate":0},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False,"dropout_rate":0},
            {"filters":64, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class ASABlendD2v3():
    def __init__(self, filters:int = 128, batch_norm:bool = True, dropout:float=0.1):
        self.name = f"asa-blend2-d2-{filters}"
        self.attention = True
        self.combine_kernel_size = 1
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
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "kernel_size":5, "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"op":"res2d", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":batch_norm, "batch_norm":False}
        ]
        self.feature_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":dropout, "batch_norm":False},
            {"filters":1., "op":"conv", "weighted_sum":False, "weight_scaled":False, "dropout_rate":0, "batch_norm":batch_norm}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":False, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":False, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]
