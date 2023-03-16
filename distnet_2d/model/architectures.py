class ASABlend():
    def __init__(self, filters:int = 192, up_filters:int=64): # up filters was 94
        self.name = f"asa-blend{filters}"
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
        if v2:
            del self.encoder_settings[2][0]
        self.feature_settings = [
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":filters, "op":"selfattention"},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
        ]
        self.feature_blending_settings = [
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
        ]

        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":True, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False,"dropout_rate":0},
            {"filters":up_filters, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]

class ASA():
    def __init__(self):
        self.name = "d3_erf_2d_asa"
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
                {"filters":64, "op":"res2d", "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
                {"filters":192, "downscale":2, "weight_scaled":False, "dropout_rate":0}
            ]
        ]
        self.feature_settings = [
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":192, "op":"selfattention"},
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":192, "op":"selfattention"},
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
            {"filters":192, "op":"res2d", "dilation":2, "weighted_sum":True, "weight_scaled":False, "dropout_rate":0},
        ]
        self.feature_blending_settings = []
        self.decoder_settings = [
            {"filters":1, "op":"conv", "n_conv":0, "conv_kernel_size":4, "up_kernel_size":4, "weight_scaled_up":False, "batch_norm_up":True, "dropout_rate":0},
            {"filters":32, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False,"dropout_rate":0},
            {"filters":94, "op":"res2d", "weighted_sum":True, "n_conv":2, "up_kernel_size":4, "weight_scaled_up":False, "weight_scaled":False, "batch_norm":False, "dropout_rate":0}
        ]
