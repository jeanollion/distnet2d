import tensorflow as tf
from tensorflow import keras
import numpy as np
from .distnet_2d import encoder_op, decoder_op, parse_param_list
from .layers import Combine, ChannelToBatch, SplitBatch
from ..utils.losses import PseudoHuber, weighted_loss_by_category, balanced_category_loss
from .gradient_accumulator import GradientAccumulator

class DistnetModelSeg(keras.Model):
    def __init__(self, *args,
                 edm_loss= PseudoHuber(1),
                 cdm_loss = PseudoHuber(1),
                 category_number: int = 0, category_class_weights=None, category_max_class_weight=10,
                 accum_steps=1,
                 cdm_loss_radius:float = 0,  # if <=0 : cdm is trained inside cells. otherwise cdm is trained only on true cdm values lower than this threhsold
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.edm_loss = edm_loss
        self.cdm_loss=cdm_loss
        self.cdm_loss_radius=float(cdm_loss_radius)
        if category_number > 1:
            if category_class_weights is not None:
                assert len(category_class_weights) == category_number, f"{category_number} category weights should be provided"
                self.category_loss = weighted_loss_by_category(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), category_class_weights, remove_background=True)
            else:
                self.category_loss = balanced_category_loss(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), category_number, max_class_frequency=category_max_class_weight, remove_background=True)
        else:
            self.category_loss = None
        self.use_grad_acc = accum_steps>1
        self.accum_steps = float(accum_steps)
        if self.use_grad_acc:
            self.gradient_accumulator = GradientAccumulator(accum_steps, self)

        # override losses reduction to None for tf.distribute.MirroredStrategy and MultiWorkerStrategy
        self.edm_loss.reduction = tf.keras.losses.Reduction.NONE
        self.cdm_loss.reduction = tf.keras.losses.Reduction.NONE

        # metrics associated to losses for to display accurate loss in a distributed setting
        self.edm_loss_metric = tf.keras.metrics.Mean(name="EDM")
        self.center_loss_metric = tf.keras.metrics.Mean(name="CDM")
        if self.category_loss is not None:
            self.category_loss_metric = tf.keras.metrics.Mean(name="category")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")


    def train_step(self, data):
        if self.use_grad_acc:
            self.gradient_accumulator.init_train_step()
        mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        x, y = data
        batch_dim = tf.shape(x)[0]

        with tf.GradientTape(persistent=False) as tape:
            y_pred = self(x, training=True)  # Forward pass
            # compute loss
            losses = dict()
            # edm
            edm_loss = self.edm_loss(y[0], y_pred[0])
            edm_loss = tf.reduce_mean(edm_loss)
            losses["EDM"] = edm_loss
            # center
            if self.cdm_loss_radius <=0: # compute loss only inside cell
                cell_mask = tf.math.greater(y[0], 0)
                center_pred_masked = tf.where(cell_mask, y_pred[1], 0) # do not compute loss outside cells
            else: # compute loss only where true CDM is lower than radius
                cell_mask = None
                center_pred_masked = tf.where(tf.math.less_equal(y[1], self.cdm_loss_radius), y_pred[1], 0)
            cdm_loss = self.cdm_loss(y[1], center_pred_masked)
            cdm_loss = tf.reduce_mean(cdm_loss)
            losses["CDM"] = cdm_loss
            if self.category_loss is not None:
                if cell_mask is None:
                    cell_mask = tf.math.greater(y[0], 0)
                cat_pred_inside = tf.where(cell_mask, y_pred[2], 1)
                cat_loss = self.category_loss(y[2], cat_pred_inside)
                cat_loss = tf.reduce_mean(cat_loss)
                losses["category"] = cat_loss
            loss = 0.
            for k, l in losses.items():
                loss += l
            if mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

            # print(f"reg loss: {len(self.losses)} values: {self.losses}")
            if len(self.losses)>0:
                loss += tf.add_n(self.losses) # regularizers
            losses["loss"] = loss

            # scale loss for distribution
            num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
            if num_replicas > 1:
                loss *= 1.0 / num_replicas

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        if mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if not self.use_grad_acc:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update weights
        else:
            self.gradient_accumulator.accumulate_gradients(gradients)
            self.gradient_accumulator.apply_gradients()

        self.edm_loss_metric.update_state(losses["EDM"], sample_weight=batch_dim)
        self.center_loss_metric.update_state(losses["CDM"], sample_weight=batch_dim)
        if self.category_loss is not None:
            self.category_loss_metric.update_state(losses["category"], sample_weight=batch_dim)
        self.loss_metric.update_state(losses["loss"], sample_weight=batch_dim)

        return self.compute_metrics(x, y, y_pred, None)

def get_distnet_2d_seg(n_inputs:int ,config, name: str="DiSTNet2DSeg",**kwargs):
    return get_distnet_2d_seg_model(n_inputs=n_inputs,
                                    upsampling_mode=config.upsampling_mode, downsampling_mode=config.downsampling_mode,
                                    skip_stop_gradient=False, skip_connections=config.skip_connections,
                                    encoder_settings=config.encoder_settings, feature_settings=config.feature_settings,
                                    decoder_settings=config.decoder_settings, feature_decoder_settings=config.feature_decoder_settings,
                                    combine_kernel_size=config.combine_kernel_size, name=name, **kwargs)

def get_distnet_2d_seg_model(n_inputs: int, # inputs are concatenated in the channel axis
                             encoder_settings: list,
                             feature_settings: list,
                             feature_decoder_settings: list,
                             decoder_settings: list,
                             spatial_dimensions:list = [None, None],
                             upsampling_mode: str = "tconv",  # tconv, up_nn, up_bilinear
                             downsampling_mode: str = "maxpool_and_stride",  # maxpool, stride, maxpool_and_stride
                             combine_kernel_size: int = 1,
                             skip_stop_gradient: bool = False,
                             skip_connections=True,  # bool or list
                             skip_combine_mode: str = "conv",  # conv, wsconv
                             scale_edm:bool = False,
                             category_number:int = 0,
                             l2_reg: float = 0,
                             name: str = "DiSTNet2DSeg",
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
    last_input_filters = n_inputs
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

    # define feature operations
    feature_convs, _, _, feature_filters, _ = parse_param_list(feature_settings, "FeatureSequence", l2_reg=l2_reg,
                                                               last_input_filters=out_filters)
    combine_features_op = Combine(filters=feature_filters, kernel_size=combine_kernel_size, compensate_gradient=True,
                                  l2_reg=l2_reg, name="CombineFeatures")
    # define decoder operations
    decoder_layers = {"Seg": [], "Center": []}
    if category_number>1:
        decoder_layers["Category"] = []
    get_seq_and_filters = lambda l: [l[i] for i in [0, 3]]
    decoder_feature_op = {n: get_seq_and_filters(
        parse_param_list(feature_decoder_settings, f"Features{n}", l2_reg=l2_reg, last_input_filters=feature_filters))
        for n in decoder_layers.keys()}
    decoder_out = {"Seg": {}, "Center": {}}
    output_per_decoder = {"Seg": ["EDM"], "Center": ["CDM"]}
    skip_per_decoder = {"Seg": skip_connections, "Center": []}
    if category_number > 1:
        decoder_out["Category"] = {}
        output_per_decoder["Category"] = ["Category"]
        skip_per_decoder["Category"] = []
    for l_idx, param_list in enumerate(decoder_settings):
        if l_idx == 0:
            decoder_out["Seg"]["EDM"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx],
                                                   mode=upsampling_mode, skip_combine_mode=skip_combine_mode,
                                                   combine_kernel_size=combine_kernel_size, activation_out="tanh" if scale_edm else "linear",
                                                   filters_out=1, l2_reg=l2_reg, layer_idx=l_idx, name=f"Output0_EDM")
            decoder_out["Center"]["CDM"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx],
                                                         mode=upsampling_mode, skip_combine_mode=skip_combine_mode,
                                                         combine_kernel_size=combine_kernel_size,
                                                         activation_out="linear", filters_out=1, l2_reg=l2_reg,
                                                         layer_idx=l_idx, name=f"Output1_CDM")
            if category_number > 1:
                decoder_out["Category"]["Category"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx],
                                                          mode=upsampling_mode, skip_combine_mode=skip_combine_mode,
                                                          combine_kernel_size=combine_kernel_size,
                                                          activation_out="softmax", filters_out=category_number, l2_reg=l2_reg,
                                                          layer_idx=l_idx, name=f"Output2_Category")
        else:
            for decoder_name, d_layers in decoder_layers.items():
                d_layers.append(decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode,
                                           skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size,
                                           activation="relu", l2_reg=l2_reg, layer_idx=l_idx,
                                           name=f"Decoder{decoder_name}"))

    # Create GRAPH
    if isinstance(spatial_dimensions, tuple):
        spatial_dimensions = list(spatial_dimensions)
    if n_inputs > 1:
        inputs = [tf.keras.layers.Input(shape=spatial_dimensions + [1], name=f"Input{i}") for i in range(n_inputs)]
        input = tf.keras.layers.Concatenate(axis=-1, name="InputConcat")(inputs)
        downsampled = [input]
    else:
        inputs = [tf.keras.layers.Input(shape=spatial_dimensions + [1], name="Input")]
        downsampled = [ inputs[0] ]

    residuals = []
    for i, l in enumerate(encoder_layers):
        down, res = l(downsampled[-1])
        downsampled.append(down)
        residuals.append(res)
    residuals = residuals[::-1]

    feature = downsampled[-1]
    for op in feature_convs:
        feature = op(feature)

    # combine individual features
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
    return DistnetModelSeg(inputs, outputs, category_number=category_number, name=name, **kwargs)
