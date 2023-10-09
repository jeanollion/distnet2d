import tensorflow as tf
from tensorflow import keras
import numpy as np
from .distnet_2d import encoder_op, decoder_op, parse_param_list
from .layers import Combine, ChannelToBatch, SplitBatch
from ..utils.losses import PseudoHuber
from .gradient_accumulator import GradientAccumulator
class DistnetModelSeg(keras.Model):
    def __init__(self, *args,
        edm_loss= PseudoHuber(1),
        gcdm_loss = PseudoHuber(1),
        accum_steps=1,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.edm_loss = edm_loss
        self.gcdm_loss=gcdm_loss
        self.use_grad_acc = accum_steps>1
        self.accum_steps = float(accum_steps)
        if self.use_grad_acc:
            self.gradient_accumulator = GradientAccumulator(accum_steps, self)

    def train_step(self, data):
        if self.use_grad_acc:
            self.gradient_accumulator.init_train_step()
        mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        x, y = data

        with tf.GradientTape(persistent=False) as tape:
            y_pred = self(x, training=True)  # Forward pass
            # compute loss
            losses = dict()
            # edm
            edm_loss = self.edm_loss(y[0], y_pred[0])
            losses["EDM"] = edm_loss
            # center
            center_pred_inside = tf.where(tf.math.greater(y[0], 0), y_pred[1], 0) # do not compute loss outside cells
            gcdm_loss = self.gcdm_loss(y[1], center_pred_inside)
            losses["GCDM"] = gcdm_loss

            loss = 0.
            for k, l in losses.items():
                loss += l
            if mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

            # print(f"reg loss: {len(self.losses)} values: {self.losses}")
            if len(self.losses)>0:
                loss += tf.add_n(self.losses) # regularizers
            losses["loss"] = loss

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        if mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if not self.use_grad_acc:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update weights
        else:
            self.gradient_accumulator.accumulate_gradients(gradients)
            self.gradient_accumulator.apply_gradients()
        return losses

def get_distnet_2d_seg(input_channels:int,config, shared_encoder:bool=False, skip_connections=None, name: str="DiSTNet2DSeg",**kwargs):
    if skip_connections is None:
        skip_connections = config.skip_connections
    return get_distnet_2d_seg_model(input_channels=input_channels, shared_encoder=shared_encoder,
                                    upsampling_mode=config.upsampling_mode, downsampling_mode=config.downsampling_mode,
                                    skip_stop_gradient=False, skip_connections=skip_connections,
                                    encoder_settings=config.encoder_settings, feature_settings=config.feature_settings,
                                    decoder_settings=config.decoder_settings, feature_decoder_settings=config.feature_decoder_settings,
                                    combine_kernel_size=config.combine_kernel_size, name=name, **kwargs)

def get_distnet_2d_seg_model(encoder_settings: list,
                             feature_settings: list,
                             feature_decoder_settings: list,
                             decoder_settings: list,
                             input_channels: int,
                             upsampling_mode: str = "tconv",  # tconv, up_nn, up_bilinear
                             downsampling_mode: str = "maxpool_and_stride",  # maxpool, stride, maxpool_and_stride
                             shared_encoder: bool = False,
                             # in case input has multiple channels -> process channels independently in encoder
                             combine_kernel_size: int = 1,
                             skip_stop_gradient: bool = False,
                             skip_connections=False,  # bool or list
                             skip_combine_mode: str = "conv",  # conv, wsconv
                             l2_reg: float = 0,
                             name: str = "DiSTNet2DSeg",
                             input_shape = [None, None],
                             **kwargs,
                             ):
    total_contraction = np.prod(
        [np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
    assert len(encoder_settings) == len(decoder_settings), "decoder should have same length as encoder"
    if skip_connections == False:
        skip_connections = []
    elif skip_connections == True:
        skip_connections = [i for i in range(len(encoder_settings) + 1)]
    else:
        assert isinstance(skip_connections, (list))
        skip_connections = [i if i >= 0 else len(encoder_settings) + 1 + i for i in skip_connections]
    # define enconder operations
    encoder_layers = []
    contraction_per_layer = []
    no_residual_layer = []
    combine_residuals = {}
    last_input_filters = 1 if shared_encoder else input_channels
    for l_idx, param_list in enumerate(encoder_settings):
        op, contraction, residual_filters, out_filters = encoder_op(param_list, downsampling_mode=downsampling_mode,
                                                                    l2_reg=l2_reg,
                                                                    skip_stop_gradient=skip_stop_gradient,
                                                                    last_input_filters=last_input_filters,
                                                                    layer_idx=l_idx)
        last_input_filters = out_filters
        encoder_layers.append(op)
        contraction_per_layer.append(contraction)
        no_residual_layer.append(residual_filters == 0)
        if shared_encoder and l_idx in skip_connections:
            combine_residuals_op = Combine(filters=residual_filters, kernel_size=combine_kernel_size,
                                           compensate_gradient=True,
                                           l2_reg=l2_reg, name=f"CombineResiduals_{l_idx}")
            combine_residuals[l_idx] = combine_residuals_op
    # define feature operations
    feature_convs, _, _, feature_filters, _ = parse_param_list(feature_settings, "FeatureSequence", l2_reg=l2_reg,
                                                               last_input_filters=out_filters)
    combine_features_op = Combine(filters=feature_filters, kernel_size=combine_kernel_size, compensate_gradient=True,
                                  l2_reg=l2_reg, name="CombineFeatures")

    if len(encoder_settings) in skip_connections:
        feature_skip_op = Combine(filters=feature_filters, l2_reg=l2_reg, name="FeatureSkip")
        feature_pair_skip_op = Combine(filters=feature_filters, l2_reg=l2_reg, name="FeaturePairSkip")

    # define decoder operations
    decoder_layers = {"Seg": [], "Center": []}
    get_seq_and_filters = lambda l: [l[i] for i in [0, 3]]
    decoder_feature_op = {n: get_seq_and_filters(
        parse_param_list(feature_decoder_settings, f"Features{n}", l2_reg=l2_reg, last_input_filters=feature_filters))
        for n in decoder_layers.keys()}
    decoder_out = {"Seg": {}, "Center": {}}
    output_per_decoder = {"Seg": ["EDM"], "Center": ["Center"]}
    skip_per_decoder = {"Seg": skip_connections, "Center": []}

    for l_idx, param_list in enumerate(decoder_settings):
        if l_idx == 0:
            decoder_out["Seg"]["EDM"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx],
                                                   mode=upsampling_mode, skip_combine_mode=skip_combine_mode,
                                                   combine_kernel_size=combine_kernel_size, activation_out="linear",
                                                   filters_out=1, l2_reg=l2_reg, layer_idx=l_idx, name=f"Output0_EDM")
            decoder_out["Center"]["Center"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx],
                                                         mode=upsampling_mode, skip_combine_mode=skip_combine_mode,
                                                         combine_kernel_size=combine_kernel_size,
                                                         activation_out="linear", filters_out=1, l2_reg=l2_reg,
                                                         layer_idx=l_idx, name=f"Output1_GCDM")
        else:
            for decoder_name, d_layers in decoder_layers.items():
                d_layers.append(decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode,
                                           skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size,
                                           activation="relu", l2_reg=l2_reg, layer_idx=l_idx,
                                           name=f"Decoder{decoder_name}"))

    # Create GRAPH
    if isinstance(input_shape, tuple):
        input_shape = list(input_shape)
    input = tf.keras.layers.Input(shape=input_shape + [input_channels], name="Input")
    print(f"input dims: {input.shape}")
    if shared_encoder:
        downsampled = [ChannelToBatch(compensate_gradient=False, name="MergeInputs")(input)]
    else:
        downsampled = [input]
    residuals = []
    for i, l in enumerate(encoder_layers):
        down, res = l(downsampled[-1])
        downsampled.append(down)
        if shared_encoder and i in skip_connections and res is not None:  # combine residuals
            res = SplitBatch(input_channels, compensate_gradient=False, name=f"SplitResiduals_{i}")(res)
            res = combine_residuals[i](res)
        residuals.append(res)
    residuals = residuals[::-1]

    feature = downsampled[-1]
    for op in feature_convs:
        feature = op(feature)

    # combine individual features
    if shared_encoder:
        all_features = SplitBatch(input_channels, compensate_gradient=False, name="SplitFeatures")(feature)
        combined_features = combine_features_op(all_features)
    else:
        combined_features = feature

    outputs = []
    for decoder_name in output_per_decoder.keys():
        d_layers = decoder_layers[decoder_name]
        skip = skip_per_decoder[decoder_name]
        up = combined_features
        for op in decoder_feature_op[decoder_name][0]:
            up = op(up)
        for i, (l, res) in enumerate(zip(d_layers[::-1], residuals[:-1])):
            up = l([up, res if len(d_layers) - 1 - i in skip else None])
        output_per_dec = dict()
        for output_name in output_per_decoder[decoder_name]:
            if output_name in decoder_out[decoder_name]:
                d_out = decoder_out[decoder_name][output_name]
                up_out = d_out([up, residuals[-1] if 0 in skip else None])  # (B, Y, X, F)
                output_per_dec[output_name] = up_out
        outputs.extend(output_per_dec.values())
    return DistnetModelSeg([input], outputs, name=name, **kwargs)
