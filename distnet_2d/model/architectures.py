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
    elif architecture_type.lower()=="TEMPY".lower():
        n_downsampling = kwargs.pop("n_downsampling", 3)
        if n_downsampling == 2:
            arch = TemPyD2
        elif n_downsampling == 3:
            arch = TemPyD3
        elif n_downsampling == 4:
            arch = TemPyD4
        else:
            raise ValueError(f"Unsupported downsampling number: {n_downsampling}: must be in [2, 3, 4]")
        return arch(**kwargs)
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")


class ArchBase:
    def __init__(self, filters:int,
                 n_inputs:int=1,
                 spatial_dimensions=[None, None],
                 frame_window:int = 3,
                 category_number: int = 0,  # category for each cell instance (segmentation level), <=1 means do not predict category
                 inference_gap_number: int = 0,
                 segmentation:bool = True,  # if false: do not output EDM and CDM
                 tracking:bool = True,  # if false: do not output dY, dX & LM
                 long_term: bool = True,
                 next: bool = True,
                 early_downsampling:bool = True,
                 scale_edm:bool = False,
                 layer_norm_dec:bool = False, layer_norm_feature_dec:bool = True, batch_norm:bool = True, dropout:float=0.2,
                 l2_reg:float=1e-4, position_encoding_l2_reg:float=1e-5,
                 downsampling_mode="maxpool_and_stride", upsampling_mode ="tconv", skip_combine_mode:str="conv",
                 attention_filters:int = 0, attention_positional_encoding:str="2d",
                 activation:str= "relu",
                 skip_connections=True, skip_stop_gradient:bool = False,
                 frame_aware:bool=False, frame_max_distance:int=0,
                 predict_fw: bool = True, predict_edm_derivatives:bool = False, predict_cdm_derivatives:bool = False,
                 ):
        assert spatial_dimensions is not None and 3 >= len(spatial_dimensions) >= 2, f"invalid spatial dimensions: {spatial_dimensions}"
        self.spatial_dimensions=spatial_dimensions
        self.tridimensional_mode=len(spatial_dimensions)==3
        self.n_inputs = n_inputs
        self.frame_window = frame_window
        self.segmentation = segmentation
        self.long_term=long_term
        self.category_number=category_number
        self.tracking=tracking
        self.inference_gap_number=inference_gap_number
        self.future_frames=next
        self.scale_edm=scale_edm
        self.skip_connections = skip_connections
        self.skip_stop_gradient=skip_stop_gradient
        self.attention_filters = attention_filters
        self.attention_positional_encoding = attention_positional_encoding
        self.self_attention = 0
        self.default_activation=activation.lower() if isinstance(activation, str) else activation
        self.downsampling_mode = downsampling_mode
        self.upsampling_mode = upsampling_mode
        self.skip_combine_mode=skip_combine_mode
        self.frame_aware=frame_aware
        self.frame_max_distance = frame_max_distance
        self.filters = filters
        self.early_downsampling = early_downsampling
        self.layer_norm_dec = layer_norm_dec
        self.layer_norm_feature_dec = layer_norm_feature_dec
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.l2_reg=l2_reg
        self.position_encoding_l2_reg=position_encoding_l2_reg
        self.predict_fw=predict_fw
        self.predict_edm_derivatives=predict_edm_derivatives
        self.predict_cdm_derivatives=predict_cdm_derivatives
        # to be defined in ArchDepth
        self.encoder_settings = None
        self.feature_settings = None
        self.decoder_settings = None
        self.feature_decoder_settings = None
        self.kernel_size_fd = None
        self.blend_combine_kernel_size = None
        self.pair_combine_kernel_size = None
        self.feature_spatial_dimensions = None
        # window size for WindowGroupNormalization; overridden by Blend / TemPy subclasses
        self.window_norm_size = 32

    def requires_input_spatial_dim(self):
        return self.self_attention > 0

class ArchDepth(ArchBase):
    def __init__(self, filters:int, **kwargs):
        super().__init__(filters, **kwargs)


class D2(ArchDepth):
    def __init__(self, pair_combine_kernel_size:int, blend_combine_kernel_size:int=1, kernel_size_fd:int=5, max_dilation:int=4, **kwargs):
        super().__init__(**kwargs)
        down_ker0 = get_downsampling_factor(2, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        down1 = down_ker0
        down_ker1 = get_downsampling_factor(2, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        ker1_2, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        down2 = spatial_contraction_product(down_ker0, down_ker1)
        ker2, dil2 = get_kernels_and_dilation(5, min(2, max_dilation), self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        ker2_2, dil2_2 = get_kernels_and_dilation(5, min(3, max_dilation), self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        ker2_3, dil2_3 = get_kernels_and_dilation(5, min(4,max_dilation), self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        self.kernel_size_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        self.blend_combine_kernel_size, _ = get_kernels_and_dilation(blend_combine_kernel_size, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        self.pair_combine_kernel_size, _ = get_kernels_and_dilation(pair_combine_kernel_size, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        self.feature_spatial_dimensions = [sd // d if sd is not None and sd > 0 else None for (sd, d) in zip(self.spatial_dimensions, ensure_multiplicity(len(self.spatial_dimensions), down2))]
        print(f"spatial dimension at feature layer: {self.feature_spatial_dimensions}")
        self.encoder_settings = [
            [
                {"filters": 32, "op": "conv", "kernel_size": ker0, "weighted_sum": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "kernel_size": ker0, "downscale": down_ker0, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "op": "conv", "kernel_size": ker1, "weighted_sum": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "op": "conv", "kernel_size": ker1_2, "weighted_sum": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": self.filters, "kernel_size": ker1, "downscale": down_ker1, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)
        self.feature_settings = [
            {"op": "resconv", "dilation": dil2, "kernel_size": ker2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil2 if self.self_attention > 0 else dil2_2,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "resconv", "attention_filters": self.attention_filters,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_3,
             "dilation": dil2 if self.self_attention > 0 else dil2_3, "dropout_rate": self.dropout,
             "num_attention_heads": self.self_attention},
            {"op": "resconv", "dilation": dil2, "kernel_size": ker2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil2 if self.self_attention > 0 else dil2_2,
             "kernel_size": ker2 if self.self_attention > 0 else ker2_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker2, "weighted_sum": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "resconv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": 0,
             "batch_norm": False, "layer_norm":self.layer_norm_feature_dec}
        ]
        self.decoder_settings = [
            {"filters": 16, "ops": [], "conv_kernel_size": ker0, "up_kernel_size": spatial_contraction_product(down_ker0, 2),
              "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 32, "ops": ["conv", "resconv"], "conv_kernel_size":ker1, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker1, 2),
              "layer_norm": [self.layer_norm_dec, False], "dropout_rate": 0}
        ]


class D3(ArchDepth):
    def __init__(self, pair_combine_kernel_size:int, blend_combine_kernel_size:int=1, kernel_size_fd:int=5, max_dilation:int=4, **kwargs):
        super().__init__(**kwargs)

        down_ker0 = get_downsampling_factor(2, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        down1 = down_ker0
        down_ker1 = get_downsampling_factor(2, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        down2 = spatial_contraction_product(down_ker0, down_ker1)
        down_ker2 = get_downsampling_factor(2, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        ker2, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        down3 = spatial_contraction_product(down_ker0, down_ker1, down_ker2)
        ker3, dil3 = get_kernels_and_dilation(5, min(2, max_dilation), self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        ker3_2, dil3_2 = get_kernels_and_dilation(5, min(3, max_dilation), self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        ker3_3, dil3_3 = get_kernels_and_dilation(5, min(4, max_dilation), self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        self.kernel_size_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        self.blend_combine_kernel_size, _ = get_kernels_and_dilation(blend_combine_kernel_size, 1, self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        self.pair_combine_kernel_size, _ = get_kernels_and_dilation(pair_combine_kernel_size, 1,  self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        self.feature_spatial_dimensions = [sd // d if sd is not None and sd > 0 else None for (sd, d) in zip(self.spatial_dimensions, ensure_multiplicity(len(self.spatial_dimensions), down3))]
        print(f"spatial dimension at feature layer: {self.feature_spatial_dimensions}")
        self.encoder_settings = [
            [
                {"filters": 32, "op": "conv", "kernel_size": ker0, "weighted_sum": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 32, "kernel_size": ker0, "downscale": down_ker0, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "kernel_size": ker1, "dropout_rate": 0},
                {"filters": 64, "kernel_size": ker1, "downscale": down_ker1, "dropout_rate": 0}
            ],
            [
                {"filters": 64, "op": "resconv", "kernel_size": ker2, "weighted_sum": False,
                 "dropout_rate": 0},
                {"filters": 64, "op": "resconv", "kernel_size": ker2, "weighted_sum": False,
                 "dropout_rate": 0},
                {"filters": self.filters, "kernel_size": ker2, "downscale": down_ker2, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)

        self.feature_settings = [
            {"op": "resconv", "dilation": dil3, "kernel_size": ker3, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil3 if self.self_attention > 0 else dil3_2,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "resconv", "attention_filters": self.attention_filters,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_3,
             "dilation": dil3 if self.self_attention > 0 else dil3_3, "dropout_rate": self.dropout},
            {"op": "resconv", "dilation": dil3, "kernel_size": ker3, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil3 if self.self_attention > 0 else dil3_2,
             "kernel_size": ker3 if self.self_attention > 0 else ker3_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker3, "weighted_sum": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "resconv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": self.kernel_size_fd, "weighted_sum": False, "dropout_rate": 0,
             "batch_norm": False, "layer_norm":self.layer_norm_feature_dec}
        ]
        self.decoder_settings = [
            {"filters": 16, "ops": [], "conv_kernel_size": ker0, "up_kernel_size": spatial_contraction_product(down_ker0, 2),
              "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 32, "ops": ["resconv"]*2, "conv_kernel_size" : ker1, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker1, 2),
              "batch_norm": False, "dropout_rate": 0},
            {"filters": 64, "ops": ["conv", "resconv"], "conv_kernel_size" : ker2, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker2, 2),
              "layer_norm": [self.layer_norm_dec, False], "dropout_rate": 0}
        ]


class D4(ArchDepth):
    def __init__(self, pair_combine_kernel_size:int, blend_combine_kernel_size:int=1, kernel_size_fd:int=5, max_dilation:int=4, **kwargs):
        super().__init__(**kwargs)
        down_ker0 = get_downsampling_factor(2, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        ker0, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, 1, tridimensional_mode=self.tridimensional_mode)
        down1 = down_ker0
        down_ker1 = get_downsampling_factor(2, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        ker1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down1, tridimensional_mode=self.tridimensional_mode)
        down2 = spatial_contraction_product(down_ker0, down_ker1)
        down_ker2 = get_downsampling_factor(2, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        ker2, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        ker2_1, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down2, tridimensional_mode=self.tridimensional_mode)
        down3 = spatial_contraction_product(down_ker0, down_ker1, down_ker2)
        down_ker3 = get_downsampling_factor(2, self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        ker3, _ = get_kernels_and_dilation(5, 1, self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        ker3_2, dil3_2 = get_kernels_and_dilation(5, min(2, max_dilation), self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        ker3_3, _ = get_kernels_and_dilation(3, 1, self.spatial_dimensions, down3, tridimensional_mode=self.tridimensional_mode)
        down4 = spatial_contraction_product(down_ker0, down_ker1, down_ker2, down_ker3)
        ker4, dil4 = get_kernels_and_dilation(5, min(2, max_dilation), self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        ker4_2, dil4_2 = get_kernels_and_dilation(5, min(3,max_dilation), self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        ker4_3, dil4_3 = get_kernels_and_dilation(5, min(4, max_dilation), self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        self.kernel_size_fd, _ = get_kernels_and_dilation(kernel_size_fd, 1, self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        self.blend_combine_kernel_size, _ = get_kernels_and_dilation(blend_combine_kernel_size, 1, self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        self.pair_combine_kernel_size, _ = get_kernels_and_dilation(pair_combine_kernel_size, 1,  self.spatial_dimensions, down4, tridimensional_mode=self.tridimensional_mode)
        self.feature_spatial_dimensions = [sd // d if sd is not None and sd > 0 else None for (sd, d) in zip(self.spatial_dimensions, ensure_multiplicity(len(self.spatial_dimensions), down4))]
        print(f"spatial dimension at feature layer: {self.feature_spatial_dimensions}")
        self.encoder_settings = [
            [
                {"filters": 16, "op": "conv", "kernel_size": ker0, "weighted_sum": False,
                 "dropout_rate": 0, "batch_norm": False},
                {"filters": 16, "kernel_size": ker0, "downscale": down_ker0, "dropout_rate": 0}
            ],
            [
                {"filters": 16, "kernel_size": ker1, "dropout_rate": 0},
                {"filters": 32, "kernel_size": ker1, "downscale": down_ker1, "dropout_rate": 0}
            ],
            [
                {"filters": 32, "op": "resconv", "kernel_size": ker2, "weighted_sum": False,
                 "dropout_rate": 0},
                {"filters": 32, "op": "resconv", "kernel_size": ker2, "weighted_sum": False,
                 "dropout_rate": 0},
                {"filters": 64, "kernel_size": ker2, "downscale": down_ker2, "dropout_rate": 0,
                 "batch_norm": False}
            ],
            [
                {"filters": 64, "op": "resconv", "kernel_size": ker3, "weighted_sum": False,
                 "dropout_rate": 0},
                {"filters": 64, "op": "resconv", "kernel_size": ker3_2, "dilation": dil3_2, "weighted_sum": False, "dropout_rate": 0},
                {"filters": self.filters, "kernel_size": ker3, "downscale": down_ker3, "dropout_rate": 0,
                 "batch_norm": False}
            ]
        ]
        if self.early_downsampling:
            self.encoder_settings[0].pop(0)

        self.feature_settings = [
            {"op": "resconv", "dilation": dil4, "kernel_size": ker4, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil4 if self.self_attention > 0 else dil4_2,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": self.filters, "op": "selfattention" if self.self_attention > 0 else "resconv", "attention_filters": self.attention_filters,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_3,
             "dilation": dil4 if self.self_attention > 0 else dil4_3, "dropout_rate": self.dropout},
            {"op": "resconv", "dilation": dil4, "kernel_size": ker4, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "dilation": dil4 if self.self_attention > 0 else dil4_2,
             "kernel_size": ker4 if self.self_attention > 0 else ker4_2, "weighted_sum": False,
             "dropout_rate": self.dropout, "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size": ker4, "weighted_sum": False,
             "dropout_rate": 0, "batch_norm": self.batch_norm},
        ]
        self.feature_decoder_settings = [
            {"filters": 0.5, "op": "conv", "kernel_size":self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "resconv", "kernel_size":self.kernel_size_fd, "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"filters": 1., "op": "conv", "kernel_size":self.kernel_size_fd, "weighted_sum": False, "dropout_rate": 0,
             "batch_norm": False, "layer_norm":self.layer_norm_feature_dec }
        ]
        self.decoder_settings = [
            {"filters": 16, "ops": [], "conv_kernel_size": ker0, "up_kernel_size": spatial_contraction_product(down_ker0, 2),
              "batch_norm_up": False, "dropout_rate": 0},
            {"filters": 16, "ops": ["resconv"]*2, "conv_kernel_size": ker1, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker1, 2),
              "batch_norm": False, "dropout_rate": 0},
            {"filters": 32, "ops": ["resconv"]*2, "conv_kernel_size": ker2_1, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker2, 2),
              "batch_norm": False, "dropout_rate": 0},
            {"filters": 64, "ops": ["conv", "resconv"], "conv_kernel_size": ker3_3, "weighted_sum": False, "up_kernel_size": spatial_contraction_product(down_ker3, 2),
              "layer_norm": [self.layer_norm_dec, False], "dropout_rate": 0}
        ]


class Blend(ArchBase):
    def __init__(self, frame_aware:bool, attention:int=0, self_attention:int=0, blending_filter_factor:float=0.5, **kwargs):
        super().__init__(frame_aware=frame_aware, batch_norm=True, layer_norm_dec=False, **kwargs)
        if attention > 0 or self_attention:
            assert self.spatial_dimensions is not None and min( self.spatial_dimensions) > 0, f"for attention mechanism, spatial dim must be provided. Got {self.spatial_dimensions}"
        self.attention = attention
        self.self_attention = self_attention
        self.blending_filter_factor = blending_filter_factor
        # window_norm_size = feature-layer spatial dim if known, else 32
        fsd = self.feature_spatial_dimensions
        if fsd is not None and any(d is not None and d > 0 for d in fsd):
            if all(d == fsd[0] for d in fsd):
                self.window_norm_size = fsd[0]
            else:
                self.window_norm_size = [s if s is not None and s>0 else 32 for s in list(fsd)]
        else:
            self.window_norm_size = 32
        self.feature_blending_settings = [
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False},
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout,
             "batch_norm": False}
        ]
    def requires_input_spatial_dim(self):
        if self.attention > 0:
            return True
        else:
            return super(Blend, self).requires_input_spatial_dim()

class BlendD2(Blend, D2):
    def __init__(self, filters:int = 128, **kwargs):
        super().__init__(filters=filters, pair_combine_kernel_size=5, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD2-{filters}"


class BlendD3(Blend, D3):
    def __init__(self, filters:int = 192, **kwargs):
        super().__init__(filters=filters, pair_combine_kernel_size=5, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD3-{filters}"


class BlendD4(Blend, D4):
    def __init__(self, filters:int = 192, **kwargs):
        super().__init__(filters=filters, pair_combine_kernel_size=5, kernel_size_fd=3, **kwargs)
        prefix = f"{'a' if self.attention else ''}{'sa' if self.self_attention else ''}"
        self.name = f"{prefix}blendD4-{filters}"


class TemPy(ArchBase):
    def __init__(self, window_attention:int, wsa_edm:bool=False, wsa_cdm:bool=False, frame_aware:bool=True, **kwargs):
        super().__init__(frame_aware=frame_aware, batch_norm=True, layer_norm_dec=False, **kwargs)
        self.window_attention = window_attention
        if self.frame_window > 0:
            assert window_attention > 0
        self.wsa_edm = wsa_edm
        self.wsa_cdm = wsa_cdm
        self.feature_blending_settings = [
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout, "batch_norm": False},
            {"op": "resconv", "weighted_sum": False, "dropout_rate": self.dropout, "batch_norm": False}
        ]
        # to be defined:
        self.attention_spatial_radius = None

    def requires_input_spatial_dim(self):
        return super(TemPy, self).requires_input_spatial_dim()

class TemPyD2(TemPy, D2):
    def __init__(self, attention_spatial_radius:int, **kwargs):
        super().__init__(pair_combine_kernel_size=1, blend_combine_kernel_size=5, max_dilation=1, **kwargs)
        self.attention_spatial_radius = limit_radius(attention_spatial_radius, self.spatial_dimensions, 2 ** 2, message="Temporal Attention")
        self.window_norm_size = self.attention_spatial_radius

class TemPyD3(TemPy, D3):
    def __init__(self, attention_spatial_radius:int, **kwargs):
        super().__init__(pair_combine_kernel_size=1, blend_combine_kernel_size=5, max_dilation=1, **kwargs)
        self.attention_spatial_radius = limit_radius(attention_spatial_radius, self.spatial_dimensions, 2 ** 3, message="Temporal Attention")
        self.window_norm_size = self.attention_spatial_radius

class TemPyD4(TemPy, D4):
    def __init__(self, attention_spatial_radius:int, **kwargs):
        super().__init__(pair_combine_kernel_size=1, blend_combine_kernel_size=5, max_dilation=1, **kwargs)
        self.attention_spatial_radius = limit_radius(attention_spatial_radius, self.spatial_dimensions, 2 ** 4, message="Temporal Attention")
        self.window_norm_size = self.attention_spatial_radius

def get_kernels_and_dilation(target_kernel, target_dilation, spa_dimensions, downsampling, tridimensional_mode:bool=False):
    ndims = 2 if not tridimensional_mode else 3
    if spa_dimensions is None:
        return target_kernel, target_dilation
    spa_dimensions = ensure_multiplicity(ndims, spa_dimensions)
    if isinstance(target_kernel, int):
        if tridimensional_mode:
            target_kernel = [min(3, target_kernel), target_kernel, target_kernel] # Z, Y, X
    kernel = ensure_multiplicity(ndims, target_kernel)
    if isinstance(target_dilation, int):
        if tridimensional_mode:
            target_dilation = [1, target_dilation, target_dilation] # Z, Y, X
    dilation = ensure_multiplicity(ndims, target_dilation)
    downsampling = ensure_multiplicity(ndims, downsampling)
    spa_dimensions = [d/ds if d is not None and d>0 else None for d, ds in zip(spa_dimensions, downsampling)]
    for i in range(len(spa_dimensions)):
        while not test_ker_dil(kernel[i], dilation[i], spa_dimensions[i]):
            if dilation[i] > 1:
                dilation[i] -=1
            elif kernel[i] > 1:
                kernel[i] = 1 + 2 * ((kernel[i] - 1) // 2 - 1)
            else:
                raise ValueError(f"Cannot find kernel size that suit dimension: {spa_dimensions[i]}")
    kernel = kernel[0] if ndims==2 and kernel[0] == kernel[1] else kernel
    dilation = dilation[0] if ndims==2 and dilation[0] == dilation[1] else dilation
    return kernel, dilation

def test_ker_dil(ker, dil, dim):
    if ker==0 or ker == 1 and dil == 1 or dim is None or dim <= 0:
        return True
    size = (ker-1)*dil
    return dim >= size * 2

def get_downsampling_factor(target_downsampling, spa_dimensions, downsampling, tridimensional_mode:bool):
    ndims = 2 if not tridimensional_mode else 3
    if spa_dimensions is None:
        return target_downsampling
    spa_dimensions = ensure_multiplicity(ndims, spa_dimensions)
    downsampling = ensure_multiplicity(ndims, downsampling)
    result_downsampling = ensure_multiplicity(ndims, target_downsampling)
    spa_dimensions = [dim / ds if dim is not None and dim > 0 else None for dim, ds in zip(spa_dimensions, downsampling)]
    for i in range(len(spa_dimensions)):
        while float(spa_dimensions[i])/float(result_downsampling[i]) < 1:
            result_downsampling[i] -= 1
    return result_downsampling[0] if ndims == 2 and result_downsampling[0] == result_downsampling[1] else result_downsampling


def spatial_contraction_product(*down):
    dim = [1 if isinstance(d, int) else len(d) for d in down]
    max_dim = max(dim)
    if max_dim == 1:
        return math.prod(down)
    down = [ensure_multiplicity(max_dim, d) for d in down]
    res=[]
    for i in range(max_dim):
        res.append(math.prod([d[i] for d in down]))
    return res

def limit_radius(target_radius, spa_dimensions, downsampling, message:str=None):
    if target_radius == 0 or spa_dimensions is None:
        return target_radius
    ndims = len(spa_dimensions)
    spa_dimensions = ensure_multiplicity(ndims, spa_dimensions)
    rad = ensure_multiplicity(ndims, target_radius)
    downsampling = ensure_multiplicity(ndims, downsampling)
    spa_dimensions = [max(1, d // ds) if d is not None and d > 0 else None for d, ds in zip(spa_dimensions, downsampling)]
    rad = [min(s, r) for r, s in zip(rad, spa_dimensions)]
    if all(r == rad[0] for r in rad):
        rad = rad[0]
    if message is not None:
        print(f"{message} rad: target={target_radius} -> actual={rad} for dim: {spa_dimensions}")
    return rad
