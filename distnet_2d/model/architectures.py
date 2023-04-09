class ASABlendD3():
    def __init__(self, filters:int = 192, batch_norm:bool = True):
        self.name = f"asa-blend-{filters}"
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
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0}
        ]
        self.feature_blending_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":batch_norm, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False,"dropout_rate":0},
            {"filters":64, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class ASABlendD2():
    def __init__(self, filters:int = 128, batch_norm:bool = True):
        self.name = f"asa-blend-d2-{filters}"
        self.attention = True
        self.combine_kernel_size = 1
        self.downsampling_mode="maxpool_and_stride"
        self.upsampling_mode ="tconv"
        self.encoder_settings = [
            [
                {"filters":32, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ],
            [
                {"filters":32, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":32, "op":"conv", "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":filters, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ]
        ]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "kernel_size":5, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0}
        ]
        self.feature_blending_decoder_settings = [
            {"filters":0.5, "op":"conv", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0}
        ]
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":batch_norm, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]
