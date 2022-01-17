import tensorflow as tf
from .layers import ConvNormAct, Bneck, Upsampling2D, StopGradient, Combine
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from .self_attention import SelfAttention
from .directional_2d_self_attention import Directional2DSelfAttention

ENCODER_SETTINGS = [
    [ # l1 = 128 -> 64
        {"filters":16},
        {"filters":16, "downscale":2}
    ],
    [  # l2 64 -> 32
        {"filters":32},
        {"filters":32, "downscale":2}
    ],
    [ # l3: 32->16
        {"filters":40, "expand_filters":120, "kernel_size":5, "SE":False},
        {"filters":40, "expand_filters":120, "activation":"hswish"},
        {"filters":80, "expand_filters":240, "activation":"hswish", "downscale":2},
    ],
    [ # l4: 16 -> 8
        {"filters":80, "expand_filters":200, "activation":"hswish"},
        {"filters":80, "expand_filters":184, "activation":"hswish"},
        {"filters":80, "expand_filters":184, "activation":"hswish"},
        {"filters":112, "expand_filters":480, "activation":"hswish"},
        {"filters":112, "expand_filters":672, "kernel_size":5, "activation":"hswish"},
        {"filters":160, "expand_filters":672, "activation":"hswish", "downscale":2}
    ]
]
FEATURE_SETTINGS = [
    {"filters":160, "expand_filters":960, "activation":"hswish"},
    {"filters":1024, "expand_filters":1024, "activation":"hswish"},
]

DECODER_SETTINGS_DS = [
    #f, s
    {"filters":16},
    {"filters":96},
    {"filters":128},
    {"filters":256}
]
DECODER_SETTINGS = [
    #f, s
    {"filters":64},
    {"filters":96},
    {"filters":128},
    {"filters":256}
]

class DiSTNet2D(Model):
    def __init__(
            self,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            skip_combine_mode:str = "conv", # conv, sum
            first_skip_mode:str = "sg", # sg, omit, None
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = None,
            output_conv_filters:int=32,
            output_conv_level = 1,
            directional_attention = False,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        super().__init__(name=name)
        self.output_conv_level=output_conv_level
        self.encoder_settings = encoder_settings
        self.feature_settings = feature_settings
        self.attention_filters=feature_settings[-1].get("filters")
        self.output_conv_filters = output_conv_filters
        if decoder_settings is None:
            decoder_settings = DECODER_SETTINGS_DS if output_conv_level==1 else DECODER_SETTINGS
        self.decoder_settings = decoder_settings
        self.directional_attention=directional_attention
        self.total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in self.encoder_settings])
        self.l2_reg = l2_reg
        self.downsampling_mode=downsampling_mode
        self.upsampling_mode=upsampling_mode
        self.skip_combine_mode=skip_combine_mode
        self.first_skip_mode=first_skip_mode
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"

    def build(self, input_shape):
        spatial_dims = input_shape[1:3]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        self.next = input_shape[-1]==3

        self.encoder_layers = [
            EncoderLayer(param_list, downsampling_mode=self.downsampling_mode, layer_idx = l_idx)
            for l_idx, param_list in enumerate(self.encoder_settings)
        ]
        self.feature_convs, _, _ = parse_param_list(self.feature_settings, "FeatureSequence")
        if self.directional_attention:
            self.self_attention = Directional2DSelfAttention(positional_encoding=True, name="SelfAttention")
        else:
            self.self_attention = SelfAttention(positional_encoding="2D", name="SelfAttention")
        self.attention_skip = Combine(filters=self.attention_filters//2, l2_reg=self.l2_reg, name="SelfAttentionSkip")
        self.decoder_layers = [DecoderLayer(**parameters, size_factor=self.encoder_layers[l_idx].total_contraction, conv_kernel_size=3, mode=self.upsampling_mode, skip_combine_mode=self.skip_combine_mode, skip_mode=self.first_skip_mode if l_idx==0 else None, activation="relu", l2_reg=self.l2_reg, use_bias=True, layer_idx=l_idx) for l_idx, parameters in enumerate(self.decoder_settings)]

        # outputs
        self.conv_edm = Conv2D(filters=3 if self.next else 2, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output0_EDM")
        # displacement
        self.conv_d = Conv2D(filters=self.output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvDist")
        self.conv_dy = Conv2D(filters=2 if self.next else 1, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output1_dy")
        self.conv_dx = Conv2D(filters=2 if self.next else 1, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output2_dx")
        # up_factor = np.prod([self.encoder_settings[-1-i] for i in range(1)])
        #self.d_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"Up_d" ), n)
        # categories
        self.conv_cat = Conv2D(filters=self.output_conv_filters, kernel_size=3, padding='same', activation="relu", name="Output3_Category")
        self.conv_catcur = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="ConvCatCur")
        #self.cat_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"_Up_cat" ), n)
        if next:
            self.conv_catnext = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output4_CategoryNext")
        super().build(input_shape)

    def call(self, input):
        residuals = []
        downsampled = [input]
        for l in self.encoder_layers:
            down, res = l(downsampled[-1])
            downsampled.append(down)
            residuals.append(res)

        feature = self.feature_convs(downsampled[-1])
        attention = self.self_attention(feature)
        feature = self.attention_skip([attention, feature])

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(self.decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        edm = self.conv_edm(upsampled[-1])

        displacement = self.conv_d(upsampled[-1-self.output_conv_level])
        dy = self.conv_dy(displacement)
        dx = self.conv_dx(displacement)
        #dy = self.d_up(dy)
        #dx = self.d_up(dx)

        categories = self.conv_cat(upsampled[-1-self.output_conv_level])
        cat  = self.conv_catcur(categories)
        #cat = self.cat_up(cat)
        if self.next:
            cat_next  = self.conv_catnext(categories)
            #cat_next = self.cat_up(cat_next)
            return edm, dy, dx, cat, cat_next
        else:
            return edm, dy, dx, cat

    def summary(self, input_shape, **kwargs):
        x = tf.keras.layers.Input(shape=input_shape)
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary(**kwargs)

class EncoderLayer(Layer):
    def __init__( self, param_list, downsampling_mode, name: str="EncoderLayer", layer_idx:int=1, ):
        super().__init__(name=f"{name}_{layer_idx}")
        maxpool = downsampling_mode=="maxpool"
        self.sequence, self.down_op, self.total_contraction = parse_param_list(param_list, ignore_stride=maxpool)
        if maxpool:
            self.down_op = MaxPool2D(pool_size=self.total_contraction, name=f"Maxpool{self.total_contraction}x{self.total_contraction}")

    def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[1]//self.total_contraction, input_shape[2]//self.total_contraction, input_shape[3]), input_shape

    def call(self, input):
        if self.sequence is None:
            res = input
        else:
            res = self.sequence(input)

        down = self.down_op(res)
        return down, res

class DecoderLayer(Layer):
    def __init__(
            self,
            filters: int,
            size_factor: int=2,
            conv_kernel_size:int=3,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            skip_combine_mode = "conv", # conv, sum
            skip_mode = "sg", # sg, omit, None
            activation: str="relu",
            l2_reg: float=1e-5,
            use_bias:bool = True,
            name: str="DecoderLayer",
            layer_idx:int=1,
        ):
        super().__init__(name=f"{name}_{layer_idx}")
        self.size_factor = size_factor
        self.up_op = Upsampling2D(filters=filters, kernel_size=size_factor, mode=mode, activation=activation, l2_reg=l2_reg, use_bias=use_bias)
        if skip_combine_mode=="conv":
            self.combine = Combine(filters=filters, l2_reg=l2_reg)
        else:
            self.combine = None
        self.skip_mode = skip_mode
        if skip_mode=="sg":
            self.stop_grad = StopGradient()
        self.conv = Conv2D(filters=filters, kernel_size=conv_kernel_size, padding='same', activation="relu")

    def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[1]*self.size_factor, input_shape[2]*self.size_factor, input_shape[3])

    def call(self, input):
        input, res = input
        up = self.up_op(input)
        if not "omit"!=self.skip_mode:
            if self.stop_grad is not None:
                res = self.stop_grad(res)
            if self.combine is not None:
                x = self.combine([up, res])
            else:
                x = up + res
        else:
            x = up
        x = self.conv(x)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0],input_shape[1] * self.size_factor, input_shape[2] * self.size_factor,input_shape[3]

def parse_param_list(param_list, ignore_stride:bool = False, name="Sequence"):
    total_contraction = 1
    if ignore_stride:
        param_list = [params.copy() for params in param_list]
        for params in param_list:
            total_contraction *= params.get("downscale", 1)
            params["downscale"] = 1
    # split into squence with no stride (for residual) and the rest of the sequence
    i = 0
    if param_list[0].get("downscale", 1)==1:
        if len(param_list)>1 and param_list[1].get("downscale", 1) == 1:
            sequence = tf.keras.Sequential(name = name)
            while i<len(param_list) and param_list[i].get("downscale", 1) == 1:
                sequence.add(parse_params(**param_list[i], name = f"{i}"))
                i+=1
        else:
            sequence = parse_params(**param_list[0], name = "0")
            i=1
    else:
        sequence=None

    if i<len(param_list):
        if i==len(param_list):
            down = parse_params(**param_list[i], name="DownOp")
            total_contraction *= param_list[i].get("downscale", 1)
        else:
            down = tf.keras.Sequential(name = "DownOp")
            for ii in range(i, len(param_list)):
                down.add(parse_params(**param_list[i], name = f"{i}"))
                total_contraction *= param_list[i].get("downscale", 1)
    else:
        down = None
    return sequence, down, total_contraction

def parse_params(filters, kernel_size = 3, expand_filters=0, SE=True, activation="relu", downscale=1, name="0"):
    if expand_filters <= 0:
        return Conv2D(filters=filters, kernel_size=kernel_size, strides = downscale, padding='same', activation=activation, name=f"Conv{kernel_size}x{kernel_size}f{filters}_{name}")
    else:
        return Bneck(
            out_channels=filters,
            exp_channels=expand_filters,
            kernel_size=kernel_size,
            stride=downscale,
            use_se=SE,
            act_layer=activation,
            skip=True,
            name=f"Bneck{kernel_size}x{kernel_size}f{expand_filters}-{filters}_{name}"
        )
