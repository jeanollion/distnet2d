import tensorflow as tf
from .layers import ConvNormAct, Bneck, UpSamplingLayer2D, StopGradient, Combine
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from .self_attention import SelfAttention
from .attention import Attention
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
    {"filters":256, "up_kernel_size":4}
]

def get_distnet_2d(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            skip_combine_mode:str = "conv", # conv, sum
            first_skip_mode:str = None, # sg, omit, None
            skip_stop_gradient:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = None,
            output_conv_filters:int=32,
            output_conv_level = 0,
            directional_attention = False,
            conv_before_edm = True,
            output_use_bias = False,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        attention_filters=feature_settings[-1].get("filters")
        if decoder_settings is None:
            decoder_settings = DECODER_SETTINGS_DS if output_conv_level==1 else DECODER_SETTINGS
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"

        spatial_dims = input_shape[:-1]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        next = input_shape[-1]==3

        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, _ = encoder_op(param_list, downsampling_mode=downsampling_mode, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)

        # define feature operations
        feature_convs, _, _, _ = parse_param_list(feature_settings, "FeatureSequence")
        if directional_attention:
            self_attention = Directional2DSelfAttention(positional_encoding=True, name="SelfAttention")
        else:
            self_attention = SelfAttention(positional_encoding="2D", name="SelfAttention")
        attention_skip_op = Combine(filters=attention_filters//2, name="FeatureSequence")

        # define decoder operations
        decoder_layers = [decoder_op(**parameters, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode=skip_combine_mode, skip_mode=first_skip_mode if l_idx==0 else ("sg" if skip_stop_gradient else None), activation="relu", layer_idx=l_idx) for l_idx, parameters in enumerate(decoder_settings)]

        # defin output operations
        if conv_before_edm:
            conv_edm = Conv2D(filters=output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvEDM")
        conv_edm_out = Conv2D(filters=3 if next else 2, kernel_size=1, padding='same', activation=None, use_bias=output_use_bias, name="Output0_EDM")
        ## displacement
        conv_d = Conv2D(filters=output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvDist")
        conv_dy = Conv2D(filters=2 if next else 1, kernel_size=1, padding='same', activation=None, use_bias=output_use_bias, name="Output1_dy")
        conv_dx = Conv2D(filters=2 if next else 1, kernel_size=1, padding='same', activation=None, use_bias=output_use_bias, name="Output2_dx")
        # up_factor = np.prod([self.encoder_settings[-1-i] for i in range(1)])
        #self.d_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"Up_d" ), n)
        # categories
        conv_cat = Conv2D(filters=output_conv_filters, kernel_size=3, padding='same', activation="relu", name="ConvCat")
        conv_catcur = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output3_Category")
        #self.cat_up = ApplyChannelWise(tf.keras.layers.Conv2DTranspose( 1, kernel_size=up_factor, strides=up_factor, padding='same', activation=None, use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l2_reg), name = n+"_Up_cat" ), n)
        if next:
            conv_catnext = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output4_CategoryNext")

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
        attention = self_attention(feature)
        feature = attention_skip_op([attention, feature])

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)
        if conv_before_edm:
            edm = conv_edm(upsampled[-1])
            edm = conv_edm_out(edm)
        else:
            edm = conv_edm_out(upsampled[-1])

        displacement = conv_d(upsampled[-1-output_conv_level])
        dy = conv_dy(displacement)
        dx = conv_dx(displacement)
        #dy = self.d_up(dy)
        #dx = self.d_up(dx)

        categories = conv_cat(upsampled[-1-output_conv_level])
        cat  = conv_catcur(categories)
        #cat = self.cat_up(cat)
        if next:
            cat_next  = conv_catnext(categories)
            #cat_next = self.cat_up(cat_next)
            outputs =  edm, dy, dx, cat, cat_next
        else:
            outputs = edm, dy, dx, cat
        return Model([input], outputs, name=name)

def get_distnet_2d_sep(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            skip_combine_mode:str = "conv", # conv, sum
            first_skip_mode:str = None, # sg, omit, None
            skip_stop_gradient:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = None,
            output_conv_filters:int=32,
            output_conv_level = 0,
            conv_before_edm = False,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        if decoder_settings is None:
            decoder_settings = DECODER_SETTINGS_DS if output_conv_level==1 else DECODER_SETTINGS
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"

        spatial_dims = input_shape[:-1]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        next = input_shape[-1]==3

        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        combine_residual_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters * (3 if next else 2), name=f"CombineResidulas{l_idx}"))
        # define feature operations
        feature_convs, _, _, attention_filters = parse_param_list(feature_settings, "FeatureSequence")
        attention_op = Attention(positional_encoding="2D", name="Attention")
        self_attention_op = Attention(positional_encoding="2D", name="SelfAttention")
        self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        combine_features_op = Combine(filters=attention_filters//2, name="CombineFeatures")
        attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers = [decoder_op(**parameters, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode=skip_combine_mode, skip_mode=first_skip_mode if l_idx==0 else ("sg" if skip_stop_gradient else None), activation="relu", layer_idx=l_idx) for l_idx, parameters in enumerate(decoder_settings)]

        # define output operations
        if conv_before_edm:
            conv_edm = Conv2D(filters=output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvEDM")
        conv_edm_out = Conv2D(filters=3 if next else 2, kernel_size=1, padding='same', activation=None, use_bias=True, name="Output0_EDM")
        ## displacement
        conv_d = Conv2D(filters=output_conv_filters, kernel_size=1, padding='same', activation="relu", name="ConvDist")
        conv_dy = Conv2D(filters=2 if next else 1, kernel_size=1, padding='same', activation=None, use_bias=True, name="Output1_dy")
        conv_dx = Conv2D(filters=2 if next else 1, kernel_size=1, padding='same', activation=None, use_bias=True, name="Output2_dx")

        # categories
        conv_cat = Conv2D(filters=output_conv_filters, kernel_size=3, padding='same', activation="relu", name="ConvCat")
        conv_catcur = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output3_Category")
        if next:
            conv_catnext = Conv2D(filters=4, kernel_size=1, padding='same', activation="softmax", name="Output4_CategoryNext")

        # Create GRAPH
        input = tf.keras.layers.Input(shape=input_shape, name="Input")
        inputs = tf.split(input, num_or_size_splits = 3 if next else 2, axis=-1)
        all_residuals = []
        all_downsampled = []
        all_features = []
        for i in inputs:
            residuals = []
            downsampled = [i]
            for l in encoder_layers:
                down, res = l(downsampled[-1])
                downsampled.append(down)
                residuals.append(res)
            all_residuals.append(residuals)
            all_downsampled.append(downsampled)
            feature = downsampled[-1]
            for op in feature_convs:
                feature = op(feature)
            sa = self_attention_op([feature, feature])
            feature = self_attention_skip_op([feature, sa])
            all_features.append(feature)
        combined_features = combine_features_op(all_features)
        attention = attention_op([all_features[0], all_features[1]])
        if next:
            attention_next= attention_op([all_features[1], all_features[2]])
            feature = attention_skip_op([attention, attention_next, combined_features])
        else:
            feature = attention_skip_op([attention, combined_features])

        residuals = []
        for l_idx in range(len(encoder_layers)):
            res = [residuals_c[l_idx] for residuals_c in all_residuals]
            combine_residual_op = combine_residual_layer[l_idx]
            residuals.append(combine_residual_op(res))

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        if conv_before_edm:
            edm = conv_edm(upsampled[-1])
            edm = conv_edm_out(edm)
        else:
            edm = conv_edm_out(upsampled[-1])

        displacement = conv_d(upsampled[-1-output_conv_level])
        dy = conv_dy(displacement)
        dx = conv_dx(displacement)
        #dy = self.d_up(dy)
        #dx = self.d_up(dx)

        categories = conv_cat(upsampled[-1-output_conv_level])
        cat  = conv_catcur(categories)
        #cat = self.cat_up(cat)
        if next:
            cat_next  = conv_catnext(categories)
            #cat_next = self.cat_up(cat_next)
            outputs =  edm, dy, dx, cat, cat_next
        else:
            outputs = edm, dy, dx, cat
        return Model([input], outputs, name=name)

def get_distnet_2d_sep_out(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            combine_kernel_size:int = 3,
            skip_stop_gradient:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = DECODER_SETTINGS,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"

        spatial_dims = input_shape[:-1]
        assert input_shape[-1] in [2, 3], "channel number should be in [2, 3]"
        next = input_shape[-1]==3

        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        combine_residual_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters * (3 if next else 2), name=f"CombineResidulas{l_idx}"))
        # define feature operations
        feature_convs, _, _, attention_filters = parse_param_list(feature_settings, "FeatureSequence")
        attention_op = Attention(positional_encoding="2D", name="Attention")
        self_attention_op = Attention(positional_encoding="2D", name="SelfAttention")
        self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        combine_features_op = Combine(filters=attention_filters//2, name="CombineFeatures")
        attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers=[]
        decoder_out = []
        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out.append( decoder_sep_op(**param_list, output_names = ["Output0_EDM"], name="DecoderEDM", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu") )
                decoder_out.append( decoder_sep_op(**param_list, output_names = ["Output1_dy", "Output2_dx"], name="DecoderDisplacement", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu") )
                cat_names = ["Output3_Category", "Output4_CategoryNext"] if next else ["Output3_Category"]
                decoder_out.append( decoder_sep2_op(**param_list, output_names = cat_names, name="DecoderCategory", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="softmax", filters_out=4) )
            else:
                decoder_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, skip_mode="sg" if skip_stop_gradient else None, activation="relu", layer_idx=l_idx) )

        # Create GRAPH
        input = tf.keras.layers.Input(shape=input_shape, name="Input")
        inputs = tf.split(input, num_or_size_splits = 3 if next else 2, axis=-1)
        all_residuals = []
        all_downsampled = []
        all_features = []
        for i in inputs:
            residuals = []
            downsampled = [i]
            for l in encoder_layers:
                down, res = l(downsampled[-1])
                downsampled.append(down)
                residuals.append(res)
            all_residuals.append(residuals)
            all_downsampled.append(downsampled)
            feature = downsampled[-1]
            for op in feature_convs:
                feature = op(feature)
            sa = self_attention_op([feature, feature])
            feature = self_attention_skip_op([feature, sa])
            all_features.append(feature)
        combined_features = combine_features_op(all_features)
        attention = attention_op([all_features[0], all_features[1]])
        if next:
            attention_next= attention_op([all_features[1], all_features[2]])
            feature = attention_skip_op([attention, attention_next, combined_features])
        else:
            feature = attention_skip_op([attention, combined_features])

        residuals = []
        for l_idx in range(len(encoder_layers)):
            res = [residuals_c[l_idx] for residuals_c in all_residuals]
            if l_idx>=1:
                combine_residual_op = combine_residual_layer[l_idx]
                residuals.append(combine_residual_op(res))
            else:
                residuals.append(res)

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        last_residuals = residuals[-1]
        [edm] = decoder_out[0]([ upsampled[-1], last_residuals ])
        dy, dx = decoder_out[1]([ upsampled[-1], last_residuals[1:] ])
        cat = decoder_out[2]([ upsampled[-1], last_residuals[1:] ])

        if next:
            cat, cat_next  = cat
            outputs =  edm, dy, dx, cat, cat_next
        else:
            [cat] = cat
            outputs = edm, dy, dx, cat
        return Model([input], outputs, name=name)

def encoder_op(param_list, downsampling_mode, name: str="EncoderLayer", layer_idx:int=1):
    name=f"{name}{layer_idx}"
    maxpool = downsampling_mode=="maxpool"
    sequence, down_sequence, total_contraction, residual_filters = parse_param_list(param_list, name, ignore_stride=maxpool)
    assert total_contraction>1, "invalid parameters: no contraction specified"
    if maxpool:
        down_sequence = [MaxPool2D(pool_size=total_contraction, name=f"{name}/Maxpool{total_contraction}x{total_contraction}")]

    def op(input):
        res = input
        if sequence is not None:
            for l in sequence:
                res=l(res)
        down = res
        for l in down_sequence:
            down = l(res)
        return down, res
    return op, total_contraction, residual_filters

def decoder_op(
            filters: int,
            size_factor: int=2,
            conv_kernel_size:int=3,
            up_kernel_size:int=0,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            skip_combine_mode = "conv", # conv, sum
            combine_kernel_size = 1,
            skip_mode = "sg", # sg, omit, None
            activation: str="relu",
            #l2_reg: float=1e-5,
            #use_bias:bool = True,
            name: str="DecoderLayer",
            layer_idx:int=1,
        ):
        name=f"{name}{layer_idx}"
        up_op = lambda x : upsampling_block(x, filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True) # l2_reg=l2_reg
        if skip_combine_mode=="conv":
            combine = Combine(name = name, filters=filters, kernel_size = combine_kernel_size) #, l2_reg=l2_reg
        else:
            combine = None
        if skip_mode=="sg":
            stop_grad = lambda x : stop_gradient(x, parent_name=name)
        conv = Conv2D(filters=filters, kernel_size=conv_kernel_size, padding='same', activation=activation, name=f"{name}/Conv{conv_kernel_size}x{conv_kernel_size}")
        def op(input):
            down, res = input
            up = up_op(down)
            if "omit"!=skip_mode:
                if skip_mode=="sg":
                    res = stop_grad(res)
                if combine is not None:
                    x = combine([up, res])
                else:
                    x = up + res
            else:
                x = up
            x = conv(x)
            return x
        return op

def decoder_sep_op(
            output_names:list,
            filters: int,
            size_factor: int=2,
            up_kernel_size:int=0,
            combine_kernel_size:int=3,
            conv_kernel_size:int=3,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            activation: str="relu",
            activation_out:str = "linear",
            filters_out:int = 1,
            #l2_reg: float=1e-5,
            #use_bias:bool = True,
            name: str="DecoderSepLayer",
            layer_idx:int=-1,
        ):
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = lambda x : upsampling_block(x, filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True) # l2_reg=l2_reg
        combine = Combine(name = name, filters=filters, kernel_size = combine_kernel_size) #, l2_reg=l2_reg
        conv_out = [Conv2D(filters=filters_out, kernel_size=conv_kernel_size, padding='same', activation=activation_out, name=f"{name}/{output_name}") for output_name in output_names]
        concat_out = [tf.keras.layers.Concatenate(axis=-1, name = output_name) for output_name in output_names]
        id_out = [tf.keras.layers.Lambda(lambda x: x, name = output_name) for output_name in output_names]
        def op(input):
            down, res_list = input
            up = up_op(down)
            x_list = [combine([up, res]) for res in res_list]
            if len(x_list)==1:
                return [id_out[i](conv_out[i](x_list[0])) for i in range(len(output_names))]
            else:
                return [concat_out[i]([conv_out[i](x) for x in x_list]) for i in range(len(output_names))]
        return op

def decoder_sep2_op(
            output_names:list,
            filters: int,
            size_factor: int=2,
            up_kernel_size:int=0,
            combine_kernel_size:int=3,
            conv_kernel_size:int=3,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            activation: str="relu",
            activation_out:str = "linear",
            filters_out:int = 1,
            #l2_reg: float=1e-5,
            #use_bias:bool = True,
            name: str="DecoderSepLayer",
            layer_idx:int=-1,
        ):
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = lambda x : upsampling_block(x, filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True) # l2_reg=l2_reg
        combine = Combine(name = name, filters=filters, kernel_size = combine_kernel_size) #, l2_reg=l2_reg
        conv_out = [Conv2D(filters=filters_out, kernel_size=conv_kernel_size, padding='same', activation=activation_out, name=output_name) for output_name in output_names]
        def op(input):
            down, res_list = input
            assert len(res_list)==len(output_names), "decoder_sep2 : expected as many outputs as residuals"
            up = up_op(down)
            return [ conv_out[i](combine([up, res])) for i, res in enumerate(res_list) ]
        return op

def upsampling_block(
            input,
            filters: int,
            parent_name:str,
            size_factor:int=2,
            kernel_size: int=0,
            mode:str="tconv", # tconv, up_nn, up_bilinera
            norm_layer:str=None,
            activation: str="relu",
            #l2_reg: float=1e-5,
            use_bias:bool = True,
            name: str="Upsampling2D",
        ):
        assert mode in ["tconv", "up_nn", "up_bilinear"], "invalid mode"
        if kernel_size<size_factor:
            kernel_size = size_factor
        if parent_name is not None and len(parent_name)>0:
            name = f"{parent_name}/{name}"
        if mode=="tconv":
            upsample = tf.keras.layers.Conv2DTranspose(
                filters,
                kernel_size=kernel_size,
                strides=size_factor,
                padding='same',
                activation=activation,
                use_bias=use_bias,
                # kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                name=f"{name}/tConv{kernel_size}x{kernel_size}",
            )
            conv=None
        else:
            interpolation = "nearest" if mode=="up_nn" else 'bilinear'
            upsample = tf.keras.layers.UpSampling2D(size=size_factor, interpolation=interpolation, name = f"{name}/Upsample{kernel_size}x{kernel_size}_{interpolation}")
            conv = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=1,
                padding='same',
                name=f"{name}/Conv{kernel_size}x{kernel_size}",
                # kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                use_bias=use_bias,
                activation=activation
            )
        x = upsample(input)
        if conv is not None:
            x = conv(x)
        return x

def stop_gradient(input, parent_name:str, name:str="StopGradient"):
    if parent_name is not None and len(parent_name)>0:
        name = f"{parent_name}/{name}"
    return tf.stop_gradient( input, name=name )

def parse_param_list(param_list, name:str, ignore_stride:bool = False):
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
            sequence = []
            while i<len(param_list) and param_list[i].get("downscale", 1) == 1:
                sequence.append(parse_params(**param_list[i], name = f"{name}/Op{i}"))
                residual_filters = param_list[i]["filters"]
                i+=1
        else:
            sequence = [parse_params(**param_list[0], name = f"{name}/Op")]
            residual_filters = param_list[0]["filters"]
            i=1
    else:
        sequence=None
        residual_filters = 0
    if i<len(param_list):
        if i==len(param_list):
            down = [parse_params(**param_list[i], name=f"{name}/DownOp")]
            total_contraction *= param_list[i].get("downscale", 1)
        else:
            down = []
            for ii in range(i, len(param_list)):
                down.append(parse_params(**param_list[i], name = f"{name}/DownOp{i}"))
                total_contraction *= param_list[i].get("downscale", 1)
    else:
        down = None
    return sequence, down, total_contraction, residual_filters

def parse_params(filters:int, kernel_size:int = 3, expand_filters:int=0, SE:bool=False, activation="relu", downscale:int=1, name:str=""):
    if expand_filters <= 0:
        return Conv2D(filters=filters, kernel_size=kernel_size, strides = downscale, padding='same', activation=activation, name=f"{name}/Conv{kernel_size}x{kernel_size}")
    else:
        return Bneck(
            out_channels=filters,
            exp_channels=expand_filters,
            kernel_size=kernel_size,
            stride=downscale,
            use_se=SE,
            act_layer=activation,
            skip=True,
            name=f"{name}/Bneck{kernel_size}x{kernel_size}f{expand_filters}"
        )
