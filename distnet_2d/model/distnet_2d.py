import tensorflow as tf
from .layers import ConvNormAct, Bneck, Downsampling2D, Upsampling2D, StopGradient, Combine
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from .self_attention import SelfAttention

class DiSTNet2D(Model):
    def __init__(
            self,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            skip_combine_mode:str = "conv", # conv, sum
            first_skip_mode:str = "sg", # sg, omit, None
            attention_filters:int = 512,
            output_conv_filters:int=32,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        super().__init__(name=name)
        self.down_settings = [
            # k   exp   out  SE      NL         s
            [ # l1 = 128 -> 64
                [ 3,  16,   16,  False,   "relu",  1 ],
                [ 3,  16,   16,  False,   "relu",  2 ],
            ],
            [  # l2 64 -> 32
                [ 3,  32,   32,  False,  "relu",    1 ],
                [ 3,  32,   32,  False,  "relu",    2 ],
            ],
            [ # l3: 32->16
                [ 5,  120,   40,  False,  "relu",    1 ],
                [ 3,  120,   40,  True,  "hswish",    1 ],
                [ 3,  240,   80,  True,  "hswish",    2 ],
            ],
            [ # l4: 16 -> 8
                [ 3,  200,   80,  True,   "hswish",  1 ],
                [ 3,  184,   80,  True,   "hswish",  1 ],
                [ 3,  184,   80,  True,   "hswish",  1 ],
                [ 3,  480,   112,  True,   "hswish",  1 ],
                [ 5,  672,   112,  True,   "hswish",  1 ],
                [ 3,  672,   160,  True,   "hswish",  2 ]
            ]

        ]
        self.feature_settings = [
            [ 5,  960,   160,  True,   "hswish",  1 ],
            [ 3,  960,   attention_filters,  True,   "hswish",  1 ]
        ]
        self.attention_filters=attention_filters
        self.output_conv_filters = output_conv_filters
        self.up_settings = [
            #f, s
            [16, 2],
            [96, 2],
            [128, 2],
            [256, 2]
        ]
        self.down_layers = [
            EncoderLayer(param_list, layer_idx = l_idx)
            for l_idx, param_list in enumerate(self.down_settings)
        ]

        self.feature_convs, _, _ = parse_param_list(self.feature_settings, "FeatureSequence")

        self.total_contraction = np.prod([np.prod([param[-1] for param in param_list]) for param_list in self.down_settings])
        self.attention_skip = Combine(filters=self.attention_filters//2, l2_reg=l2_reg, name="SelfAttentionSkip")

        self.up_layers = [DecoderLayer(filters, size_factor=size_factor, conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode=skip_combine_mode, skip_mode=first_skip_mode if l_idx==0 else None, activation="relu", l2_reg=l2_reg, use_bias=True, layer_idx=l_idx) for l_idx, (filters, size_factor) in enumerate(self.up_settings)]

        self.conv_d = Conv2D(filters=self.output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvDist")
        # up_factor = np.prod([self.down_settings[-1-i] for i in range(1)])
        #self.d_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"Up_d" ), n)
        self.conv_cat = Conv2D(filters=self.output_conv_filters, kernel_size=3, padding='same', activation="relu", name="Output3_Category")


        self.conv_catcur = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="ConvCatCur")

        #self.cat_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"_Up_cat" ), n)

    def build(self, input_shape):
        spatial_dims = input_shape[1:3]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        self.next = input_shape[-1]==3
        self.self_attention = SelfAttention(positional_encoding="2D", name="SelfAttention")
        self.conv_dy = Conv2D(filters=2 if self.next else 1, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output1_dy")
        self.conv_dx = Conv2D(filters=2 if self.next else 1, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output2_dx")
        self.conv_edm = Conv2D(filters=3 if self.next else 2, kernel_size=1, padding='same', activation=None, use_bias=False, name="Output0_EDM")
        if next:
            self.conv_catnext = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output4_CategoryNext")
        super().build(input_shape)

    def call(self, input):
        residuals = []
        downsampled = [input]
        for l in self.down_layers:
            down, res = l(downsampled[-1])
            downsampled.append(down)
            residuals.append(res)

        feature = self.feature_convs(downsampled[-1])
        attention, _ = self.self_attention(feature)
        feature = self.attention_skip([attention, feature])

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(self.up_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        edm = self.conv_edm(upsampled[-1])

        displacement = self.conv_d(upsampled[-2])
        dy = self.conv_dy(displacement)
        dx = self.conv_dx(displacement)
        #dy = self.d_up(dy)
        #dx = self.d_up(dx)

        categories = self.conv_cat(upsampled[-2])
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
    def __init__( self, param_list, name: str="EncoderLayer", layer_idx:int=1, ):
        super().__init__(name=f"{name}_{layer_idx}")
        self.sequence, self.down_op, self.total_contraction = parse_param_list(param_list)

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

def parse_param_list(param_list, name="Sequence"):
    # split into squence with no stride (for residual) and the rest of the sequence
    i = 0
    if param_list[0][-1]==1:
        if len(param_list)>1 and param_list[1][-1] == 1:
            sequence = tf.keras.Sequential(name = name)
            while i<len(param_list) and param_list[i][-1] == 1:
                sequence.add(parse_params(*param_list[i], name = f"{i}"))
                i+=1
        else:
            sequence = parse_params(*param_list[0], name = "0")
            i=1
    else:
        sequence=None
    total_contraction = 1
    if i<len(param_list):
        if i==len(param_list):
            down = parse_params(*param_list[i], name="DownOp")
            total_contraction *= param_list[i][-1]
        else:
            down = tf.keras.Sequential(name = "DownOp")
            for ii in range(i, len(param_list)):
                down.add(parse_params(*param_list[i], name = f"{i}"))
                total_contraction *= param_list[i][-1]
    else:
        down = None
    return sequence, down, total_contraction

def parse_params(k, exp_filters, filters, SE, act, size_factor, name):
    if exp_filters == filters:
        return Conv2D(filters=filters, kernel_size=k, strides = size_factor, padding='same', activation="relu", name=f"Conv{k}x{k}_{name}")
    else:
        return Bneck(
            out_channels=filters,
            exp_channels=exp_filters,
            kernel_size=k,
            stride=size_factor,
            use_se=SE,
            act_layer=act,
            skip=True,
            name=f"Bneck{k}x{k}_{name}"
        )
