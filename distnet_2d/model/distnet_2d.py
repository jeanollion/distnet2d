import tensorflow as tf
from .layers import ker_size_to_string, Combine, ResConv2D, Conv2DBNDrop, Conv2DTransposeBNDrop, WSConv2D, \
    BatchToChannel, SplitBatch, ChannelToBatch, NConvToBatch2D, SelectFeature, StopGradient, Stack
import numpy as np
from .spatial_attention import SpatialAttention2D
from ..utils.helpers import ensure_multiplicity, flatten_list
from ..utils.losses import weighted_loss_by_category, balanced_category_loss, PseudoHuber, compute_loss_derivatives
from ..utils.agc import adaptive_clip_grad
from .gradient_accumulator import GradientAccumulator
import time

class DiSTNetModel(tf.keras.Model):
    def __init__(self, *args, spatial_dims,
                 edm_loss_weight:float=1,
                 edm_frequency_weights:list = None, # weights to balance foreground/background classes
                 center_loss_weight:float=1,
                 displacement_loss_weight:float=1,
                 link_multiplicity_loss_weight:float=1,
                 category_loss_weight: float = 1,
                 edm_loss=PseudoHuber(1), edm_derivative_loss:bool=False,
                 cdm_loss=PseudoHuber(1), cdm_derivative_loss:bool=False,
                 cdm_loss_radius:float = 0,
                 displacement_loss=PseudoHuber(1),
                 link_multiplicity_class_weights=None,  # array of weights: [single, multiple, null] or None = auto
                 link_multiplicity_max_class_weight=50,
                 next:bool=True,
                 frame_window=3,
                 long_term:bool=True,
                 predict_next_displacement:bool=True,
                 predict_cdm_derivatives:bool=False, predict_edm_derivatives:bool=False,
                 category_number:int=0, category_class_weights = None, category_max_class_weight=10,
                 print_gradients:bool=False,  # for optimization, available in eager mode only
                 accum_steps=1, use_agc=False, agc_clip_factor=0.1, agc_eps=1e-3, agc_exclude_output=False,  # lower clip factor clips more
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.edm_weight = edm_loss_weight
        if edm_frequency_weights is not None:
            assert len(edm_frequency_weights) == 2 , "edm_frequency_weights must be a list of len 2"
        self.edm_frequency_weights = np.asarray(edm_frequency_weights, dtype="float32") if edm_frequency_weights is not None else None
        self.center_weight = center_loss_weight
        self.cdm_loss_radius = float(cdm_loss_radius)
        self.displacement_weight = displacement_loss_weight
        self.link_multiplicity_weight = link_multiplicity_loss_weight
        self.category_weight = category_loss_weight if category_number > 1 else 0
        self.category_number=category_number
        self.spatial_dims = spatial_dims
        self.next = next
        self.predict_next_displacement=predict_next_displacement
        self.frame_window = frame_window
        self.edm_loss = edm_loss
        self.edm_derivative_loss = edm_derivative_loss
        self.cdm_loss = cdm_loss
        self.cdm_derivative_loss = cdm_derivative_loss
        self.displacement_loss = displacement_loss
        self.predict_cdm_derivatives = predict_cdm_derivatives
        self.predict_edm_derivatives = predict_edm_derivatives

        if link_multiplicity_class_weights is not None:
            assert len(link_multiplicity_class_weights) == 3, "3 link multiplicity class weights should be provided: normal cell, dividing/merging cells, cell with no previous cell"
            self.link_multiplicity_loss=weighted_loss_by_category(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), link_multiplicity_class_weights, remove_background=True)
        else:
            self.link_multiplicity_loss = balanced_category_loss(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), 3, max_class_frequency=link_multiplicity_max_class_weight, remove_background=True)
        if category_number > 1:
            if category_class_weights is not None:
                assert len(category_class_weights) == category_number, f"{category_number} category weights should be provided {len(category_class_weights)} where provided instead ({category_class_weights})"
                self.category_loss = weighted_loss_by_category(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), category_class_weights, remove_background=True)
            else:
                self.category_loss = balanced_category_loss(tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE), category_number, max_class_frequency=category_max_class_weight, remove_background=True)
        else:
            self.category_loss = None
        # gradient accumulation from https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/accumulators.py
        self.long_term = long_term
        self.use_grad_acc = accum_steps>1
        self.accum_steps = float(accum_steps)
        if self.use_grad_acc or use_agc:
            self.gradient_accumulator = GradientAccumulator(accum_steps, self)
        self.use_agc = use_agc
        self.agc_clip_factor = agc_clip_factor
        self.agc_eps = agc_eps
        self.agc_exclude_keywords=["DecoderTrackY0_", "DecoderTrackX0_", "DecoderLinkMultiplicity0_", "DecoderCenterCDM0_", "DecoderCenterCDMdY0_", "DecoderCenterCDMdX0_", "DecoderSegEDM0_", "DecoderSegEDMdY0_", "DecoderSegEDMdX0_"] if agc_exclude_output else None
        self.print_gradients=print_gradients

        # override losses reduction to None for tf.distribute.MirroredStrategy and MultiWorkerStrategy

        self.edm_loss.reduction = tf.keras.losses.Reduction.NONE
        self.cdm_loss.reduction = tf.keras.losses.Reduction.NONE
        self.displacement_loss.reduction = tf.keras.losses.Reduction.NONE

        # metrics associated to losses for to display accurate loss in a distributed setting
        self.edm_loss_metric = tf.keras.metrics.Mean(name="EDM")
        self.center_loss_metric = tf.keras.metrics.Mean(name="CDM")
        self.category_loss_metric = tf.keras.metrics.Mean(name="category") if self.category_weight > 0 else None
        self.dx_loss_metric = tf.keras.metrics.Mean(name="dX")
        self.dy_loss_metric = tf.keras.metrics.Mean(name="dY")
        self.link_multiplicity_loss_metric = tf.keras.metrics.Mean(name="link_multiplicity")
        self.loss_metric = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        metrics = []
        if self.edm_weight > 0:
            metrics.append(self.edm_loss_metric)
        if self.center_weight > 0:
            metrics.append(self.center_loss_metric)
        if self.displacement_weight > 0:
            metrics.append(self.dy_loss_metric)
            metrics.append(self.dx_loss_metric)
        if self.link_multiplicity_weight > 0:
            metrics.append(self.link_multiplicity_loss_metric)
        if self.category_weight > 0:
            metrics.append(self.category_loss_metric)
        metrics.append(self.loss_metric)
        if self._is_compiled:
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics

        for l in self._flatten_layers():
            metrics.extend(l._metrics)

        return metrics


    def train_step(self, data):
        if self.use_grad_acc:
            self.gradient_accumulator.init_train_step()

        fw = self.frame_window
        n_frames = fw * (2 if self.next else 1) + 1
        n_frame_pairs = fw * (2 if self.next else 1)
        n_fp_mul = 2 if self.predict_next_displacement else 1
        if self.long_term:
            n_frame_pairs += (fw - 1) * (2 if self.next else 1)
        mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        x, y = data
        batch_dim = tf.shape(x)[0]
        displacement_weight = self.displacement_weight / (2. * n_fp_mul) if fw > 0 else 0 # y & x
        link_multiplicity_weight = self.link_multiplicity_weight / (n_fp_mul * float(n_frame_pairs)) if fw > 0 else 0
        edm_weight = self.edm_weight / float(n_frames) # divide by channel number ?
        center_weight = self.center_weight / float(n_frames)# divide by channel number ?
        category_weight = self.category_weight / float(n_frames)
        expected_outputs = 1 + int(center_weight>0)+ 2*int(displacement_weight>0) + int(link_multiplicity_weight>0) + int(category_weight>0)
        assert len(y) == expected_outputs , f"invalid number of output. Expected: {expected_outputs} actual {len(y)}" # 0 = edm, 1 = center, 2 = dY, 3 = dX, 4 = LinkMultiplicity, 5=category

        lm_idx = int(center_weight > 0) + 2 * int(displacement_weight > 0) + int(link_multiplicity_weight > 0)
        cat_idx = lm_idx + 1

        with tf.GradientTape(persistent=self.print_gradients) as tape:
            y_pred = self(x, training=True)  # Forward pass
            if self.predict_edm_derivatives:
                edm, edm_dy, edm_dx = tf.split(y_pred[0], num_or_size_splits=3, axis=-1)
            else:
                edm, edm_dy, edm_dx = y_pred[0], None, None
            if self.predict_edm_derivatives or self.edm_derivative_loss:
                true_edm, true_edm_dy, true_edm_dx = tf.split(y[0], num_or_size_splits=3, axis=-1)
            else:
                true_edm, true_edm_dy, true_edm_dx = y[0], None, None
            if center_weight > 0:
                if self.predict_cdm_derivatives:
                    cdm, cdm_dy, cdm_dx = tf.split(y_pred[1], num_or_size_splits=3, axis=-1)
                else:
                    cdm, cdm_dy, cdm_dx = y_pred[1], None, None

            # compute loss
            losses = dict()
            loss_weights = dict()

            cell_mask = tf.math.greater(true_edm, 0)
            cell_mask_interior = tf.math.greater(true_edm, 1.5) if self.cdm_derivative_loss or self.predict_cdm_derivatives or self.edm_derivative_loss else None
            # edm
            if edm_weight>0:
                weight_map = tf.where(cell_mask, self.edm_frequency_weights[1], self.edm_frequency_weights[0]) if self.edm_frequency_weights is not None else None
                edm_loss = compute_loss_derivatives(true_edm, edm, self.edm_loss, true_dy=true_edm_dy, true_dx=true_edm_dx, pred_dy=edm_dy, pred_dx=edm_dx, mask_interior=cell_mask_interior, derivative_loss=self.edm_derivative_loss, laplacian_loss=self.edm_derivative_loss, weight_map=weight_map)
                edm_loss = tf.reduce_mean(edm_loss)
                losses["EDM"] = edm_loss
                loss_weights["EDM"] = edm_weight

            # center
            if center_weight>0:
                if self.cdm_loss_radius <= 0:
                    cdm_mask = cell_mask
                    cdm_mask_interior = cell_mask_interior
                else:
                    cdm_mask = tf.math.less_equal(y[1], self.cdm_loss_radius)
                    cdm_mask_interior = cdm_mask
                center_loss = compute_loss_derivatives(y[1], cdm, self.cdm_loss, pred_dy=cdm_dy, pred_dx=cdm_dx, mask=cdm_mask, mask_interior=cdm_mask_interior, derivative_loss=self.cdm_derivative_loss)
                center_loss = tf.reduce_mean(center_loss)
                losses["CDM"] = center_loss
                loss_weights["CDM"] = center_weight

            if category_weight > 0:
                cat_loss = self._compute_category_loss(y[cat_idx], y_pred[cat_idx], cell_mask, n_frames)
                cat_loss = tf.reduce_mean(cat_loss)
                losses["category"] = cat_loss
                loss_weights["category"] = category_weight

            # regression displacement loss
            if displacement_weight > 0:
                loss_dY, loss_dX = self._compute_displacement_loss(y, y_pred, cell_mask)
                loss_dY = tf.reduce_mean(loss_dY)
                losses["dY"] = loss_dY
                loss_weights["dY"] = displacement_weight
                loss_dX = tf.reduce_mean(loss_dX)
                losses["dX"] = loss_dX
                loss_weights["dX"] = displacement_weight

            # link_multiplicity loss
            if link_multiplicity_weight>0:
                link_multiplicity_loss = self._compute_link_multiplicity_loss(y[lm_idx], y_pred[lm_idx], n_frame_pairs, n_fp_mul)
                link_multiplicity_loss = tf.reduce_mean(link_multiplicity_loss)
                losses["link_multiplicity"] = link_multiplicity_loss
                loss_weights["link_multiplicity"] = link_multiplicity_weight

            loss = 0.
            for k, l in losses.items():
                loss += l * loss_weights[k]
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

        if self.print_gradients:
            trainable_vars_tape = [t for t in self.trainable_variables if (t.name.startswith("DecoderSegEDM") or t.name.startswith("DecoderCenterCDM") or t.name.startswith("DecoderTrackY0") or t.name.startswith("DecoderTrackX0") or t.name.startswith("DecoderLinkMultiplicity0") or t.name.startswith("FeatureSequence_Op4") or t.name.startswith("Attention")) and ("/kernel" in t.name or "/wv" in t.name) ]
            for loss_name, loss_value in losses.items():
                if loss_name != "loss" :
                    w = loss_weights[loss_name] # outside tape: cannot modify loss_value -> need to apply w to gradient itself
                    if mixed_precision:
                        loss_value = self.optimizer.get_scaled_loss(loss_value)
                    gradients = tape.gradient(loss_value, trainable_vars_tape)
                    if mixed_precision:
                        gradients = self.optimizer.get_unscaled_gradients(gradients)
                    for v, g in zip(trainable_vars_tape, gradients):
                        if g is not None:
                            g = g * w
                            print(f"{v.name}, loss: {loss_name}, val: {loss_value}, grad: {tf.math.sqrt(tf.reduce_mean(tf.math.square(g))).numpy()} shape: {g.shape}")
                    if self.use_agc:
                        gradients = adaptive_clip_grad(trainable_vars_tape, gradients, clip_factor=self.agc_clip_factor, eps=self.agc_eps, exclude_keywords=self.agc_exclude_keywords, grad_scale=w)
                        for v, g in zip(trainable_vars_tape, gradients):
                            if g is not None:
                                print(f"AGC: layer: {v.name}, loss: {loss_name}, grad: {tf.math.sqrt(tf.reduce_mean(tf.math.square(g))).numpy()}")

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        if mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        if self.use_agc:
            gradients = adaptive_clip_grad(self.trainable_variables, gradients, clip_factor=self.agc_clip_factor, eps=self.agc_eps, exclude_keywords=self.agc_exclude_keywords)
        if not self.use_grad_acc:
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update weights
        else:
            self.gradient_accumulator.accumulate_gradients(gradients)
            self.gradient_accumulator.apply_gradients()

        # Update metrics state
        if edm_weight > 0:
            self.edm_loss_metric.update_state(losses["EDM"], sample_weight=batch_dim)
        if center_weight > 0:
            self.center_loss_metric.update_state(losses["CDM"], sample_weight=batch_dim)
        if displacement_weight>0:
            self.dx_loss_metric.update_state(losses["dX"], sample_weight=batch_dim)
            self.dy_loss_metric.update_state(losses["dY"], sample_weight=batch_dim)
        if link_multiplicity_weight >0:
            self.link_multiplicity_loss_metric.update_state(losses["link_multiplicity"], sample_weight=batch_dim)
        if self.category_loss_metric is not None:
            self.category_loss_metric.update_state(losses["category"], sample_weight=batch_dim)
        self.loss_metric.update_state(losses["loss"], sample_weight=batch_dim)

        return self.compute_metrics(x, y, y_pred, None)

    def _compute_displacement_loss(self, y, y_pred, cell_mask):
        idx = int(self.center_weight > 0) + 1
        mask = self._to_pair_mask(cell_mask)
        dy = tf.where(mask, y_pred[idx], 0)  # do not predict anything outside
        dx = tf.where(mask, y_pred[idx+1], 0)  # do not predict anything outside
        return self.displacement_loss(y[idx], dy), self.displacement_loss(y[idx+1], dx)

    def _to_pair_mask(self, cell_mask):
        fw = self.frame_window
        mask = cell_mask[..., 1:]
        if self.predict_next_displacement:
            mask_next = cell_mask[..., :-1]
        if self.long_term and fw > 1:
            mask_center = tf.tile(mask[..., fw - 1:fw], [1, 1, 1, fw - 1])
            if self.predict_next_displacement:
                if self.next:
                    mask = tf.concat(
                        [mask, mask_center, cell_mask[..., -fw + 1:], mask_next, cell_mask[..., :fw - 1], mask_center],
                        -1)
                else:
                    mask = tf.concat([mask, mask_center, mask_next, cell_mask[..., :fw - 1]], -1)
            else:
                if self.next:
                    mask = tf.concat([mask, mask_center, cell_mask[..., -fw + 1:]], -1)
                else:
                    mask = tf.concat([mask, mask_center], -1)
        elif self.predict_next_displacement:
            mask = tf.concat([mask, mask_next], -1)
        return mask

    def _compute_category_loss(self, y, y_pred, cell_mask, n_frames): # TODO use split instead of loop
        cn = self.category_number
        lm_loss = 0.
        for i in range(n_frames):
            cat_pred_inside = tf.where(cell_mask[..., i:i + 1], y_pred[..., cn * i:cn * i + cn], 1)
            lm_loss = lm_loss + self.category_loss(y[..., i:i + 1], cat_pred_inside)
        return lm_loss

    def _compute_link_multiplicity_loss(self, y, y_pred, n_frame_pairs, n_fp_mul): # TODO use split instead of loop
        lm_loss = 0.
        for i in range(n_frame_pairs * n_fp_mul):
            inside_mask = tf.math.greater(y[..., i:i + 1], 0)
            lm_pred_inside = tf.where(inside_mask, y_pred[..., 3 * i:3 * i + 3], 1)
            lm_loss = lm_loss + self.link_multiplicity_loss(y[..., i:i + 1], lm_pred_inside)
        return lm_loss

    def set_inference(self, inference:bool=True):
        for layer in self.layers:
            if isinstance(layer, (NConvToBatch2D, BatchToChannel, SelectFeature)):
                layer.inference_mode = inference

    def save(self, *args, inference:bool, **kwargs):
        if inference:
            self.set_inference(True)
            self.trainable=False
            self.compile()
        super().save(*args, **kwargs)
        if inference:
            self.set_inference(False)
            self.trainable=True
            self.compile()


def get_distnet_2d(spatial_dimensions, n_inputs:int,
            frame_window:int,
            next:bool,
            config,
            name: str="DiSTNet2D",
            **kwargs):
    return get_distnet_2d_model(spatial_dimensions, n_inputs, upsampling_mode=config.upsampling_mode, downsampling_mode=config.downsampling_mode, skip_connections=config.skip_connections, encoder_settings=config.encoder_settings, feature_settings=config.feature_settings, feature_blending_settings=config.feature_blending_settings, decoder_settings=config.decoder_settings, feature_decoder_settings=config.feature_decoder_settings, attention=config.attention, attention_dropout=config.dropout, self_attention=config.self_attention, attention_positional_encoding=config.attention_positional_encoding, combine_kernel_size=config.combine_kernel_size, pair_combine_kernel_size=config.pair_combine_kernel_size, blending_filter_factor=config.blending_filter_factor, frame_window=frame_window, next=next, name=name, **kwargs)

def get_distnet_2d_model(spatial_dimensions:[list, tuple],  # (Y, X)
                         n_inputs:int,
                         encoder_settings:list,
                         feature_settings: list,
                         feature_blending_settings: list,
                         feature_decoder_settings:list,
                         decoder_settings: list,
                         upsampling_mode:str="tconv",  # tconv, up_nn, up_bilinear
                         downsampling_mode:str = "maxpool_and_stride",  #maxpool, stride, maxpool_and_stride
                         combine_kernel_size:int = 1,
                         pair_combine_kernel_size:int = 1,
                         blending_filter_factor:float = 0.5,
                         skip_stop_gradient:bool = False, # was True
                         skip_connections = True,  # bool or list
                         skip_combine_mode:str="conv",  #conv, wsconv
                         attention : int = 0,
                         attention_dropout:float = 0.1,
                         attention_positional_encoding="2d",
                         self_attention: int = 0,
                         frame_window:int = 1,
                         next:bool=True,
                         long_term:bool = True,
                         tracking:bool = True, # if false: outputs are only EDM and CDM
                         inference_gap_number:int = 0,
                         predict_next_displacement:bool = True,
                         predict_edm_derivatives:bool = False,
                         predict_cdm_derivatives:bool = False,
                         scale_edm:bool = False,
                         category_number:int = 0,  # category for each cell instance (segmentation level), <=1 means do not predict category
                         l2_reg:float = 0,
                         name: str="DiSTNet2D",
                         legacy_multi_input_arch:bool=False,
                         **kwargs,
                         ):
        if frame_window == 0:
            attention = 0
            long_term = False
            tracking = False
        if not tracking:
            inference_gap_number = 0
            kwargs["displacement_loss_weight"] = 0
            kwargs["link_multiplicity_loss_weight"] = 0

        print(f"edm activation: {'tanh' if scale_edm else 'linear'}")
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        if spatial_dimensions is None:
            spatial_dimensions = [None, None]
        else:
            spatial_dimensions = list(spatial_dimensions)
            assert len(spatial_dimensions) == 2, "2D input required"
        if attention>0 or self_attention>0:
            assert spatial_dimensions[0] is not None and spatial_dimensions[0] > 0, "for attention mechanism, spatial dim must be provided"
            assert spatial_dimensions[1] is not None and spatial_dimensions[1] > 0, "for attention mechanism, spatial dim must be provided"
            print(f"attention positional encoding mode: {attention_positional_encoding}")
        else:
            spatial_dimensions = [None, None] # no attention : no need to enforce fixed size
        if frame_window<=1:
            long_term = False
        n_frames = frame_window * (2 if next else 1) + 1
        if skip_connections == False:
            skip_connections = [len(encoder_settings)] # only at feature level
        elif skip_connections == True:
            skip_connections = [i for i in range(len(encoder_settings)+1)]
        else:
            assert isinstance(skip_connections, (list))
            skip_connections = [i if i>=0 else len(encoder_settings)+1+i for i in skip_connections]
        inference_pair_idx = [frame_window - 1, frame_window]
        inference_pair_sel_bw = [0]
        inference_pair_sel_fw = [1]
        if inference_gap_number > 1:
            assert long_term, "long term must be enabled for gap prediction"
            assert inference_gap_number < frame_window, f"gap number must be lower or equal to: {frame_window-1} got {inference_gap_number}"
            n_gap_max = frame_window - 1
            n_pairs_0 = n_frames - 1
            for gap in range(inference_gap_number):
                inference_pair_sel_bw.append(len(inference_pair_idx))
                inference_pair_idx.append(n_pairs_0 + n_gap_max - gap - 1)
                if next:
                    inference_pair_sel_fw.append(len(inference_pair_idx))
                    inference_pair_idx.append(n_pairs_0 + n_gap_max + gap )
        #print(f"inference_pair_idx {inference_pair_idx} bw: {inference_pair_sel_bw}={[inference_pair_idx[i] for i in inference_pair_sel_bw]} fw: {inference_pair_sel_fw}=={[inference_pair_idx[i] for i in inference_pair_sel_fw]}")
        # define encoder operations
        encoder_layers = []
        contraction_per_layer = []
        no_residual_layer = []
        last_input_filters = n_inputs
        if n_inputs > 1 and legacy_multi_input_arch:
            last_input_filters = 1
            print(f"legacy multi-input arch")
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters, out_filters = encoder_op(param_list, skip_parameters=(n_frames, frame_window), downsampling_mode=downsampling_mode, attention_positional_encoding=attention_positional_encoding, l2_reg=l2_reg, skip_stop_gradient=skip_stop_gradient, last_input_filters = last_input_filters, layer_idx = l_idx)
            last_input_filters = out_filters
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            no_residual_layer.append(residual_filters==0)
        # define feature operations
        feature_convs, _, _, feature_filters, _ = parse_param_list(feature_settings, "FeatureSequence", attention_positional_encoding=attention_positional_encoding, l2_reg=l2_reg, last_input_filters=out_filters)
        combine_filters = int(feature_filters * n_frames  * blending_filter_factor)
        print(f"feature filters: {feature_filters} combine filters: {combine_filters}")
        combine_features_op = Combine(filters=combine_filters, kernel_size=combine_kernel_size, compensate_gradient = True, l2_reg=l2_reg, name="CombineFeatures") if frame_window > 0 else lambda features : features[0]
        if attention>0:
            attention_op = SpatialAttention2D(num_heads=attention, positional_encoding=attention_positional_encoding, dropout=attention_dropout, l2_reg=l2_reg, name="Attention")
        pair_combine_op = Combine(filters=feature_filters, kernel_size = pair_combine_kernel_size, l2_reg=l2_reg, name="FeaturePairCombine")
        all_pair_combine_op = Combine(filters=combine_filters, kernel_size=combine_kernel_size, compensate_gradient = True, l2_reg=l2_reg, name="AllFeaturePairCombine")
        feature_pair_feature_combine_op = Combine(filters=combine_filters, kernel_size=combine_kernel_size, l2_reg=l2_reg, name="FeaturePairFeatureCombine") # change here was feature_filters

        for f in feature_blending_settings:
            if "filters" not in f or f["filters"]<0:
                f["filters"] = combine_filters
        feature_blending_convs, _, _, feature_blending_filters, _ = parse_param_list(feature_blending_settings, "FeatureBlendingSequence", attention_positional_encoding=attention_positional_encoding, l2_reg=l2_reg, last_input_filters=combine_filters)

        if len(encoder_settings) in skip_connections:
            feature_skip_op = Combine(filters=feature_filters, l2_reg=l2_reg, name="FeatureSkip")
            feature_pair_skip_op = Combine(filters=feature_filters, l2_reg=l2_reg, name="FeaturePairSkip")

        # define decoder operations
        decoder_layers={"Seg":[], "Center":[], "Track":[], "LinkMultiplicity":[]} if tracking else {"Seg":[], "Center":[]}
        if category_number > 1:
            decoder_layers["Cat"] = []
        get_seq_and_filters = lambda l : [l[i] for i in [0, 3]]
        decoder_feature_op={n: get_seq_and_filters(parse_param_list(feature_decoder_settings, f"Features{n}", attention_positional_encoding=attention_positional_encoding, l2_reg=l2_reg, last_input_filters=feature_filters)) for n in decoder_layers.keys()}
        decoder_out={name:{} for name in decoder_layers.keys()}
        output_per_decoder = {"Seg": {"EDM":0}, "Center": {"CDM":1}, "Track": {"dYBW":2, "dXBW":3} if not predict_next_displacement else {"dYBW":2, "dXBW":3, "dYFW":2, "dXFW":3}, "LinkMultiplicity": {"LinkMultiplicityBW":4} if not predict_next_displacement else {"LinkMultiplicityBW":4, "LinkMultiplicityFW":4}}
        if predict_edm_derivatives:
            output_per_decoder["Seg"]["EDMdY"]=0
            output_per_decoder["Seg"]["EDMdX"]=0
        if predict_cdm_derivatives:
            output_per_decoder["Center"]["CDMdY"]=1
            output_per_decoder["Center"]["CDMdX"]=1
        if category_number > 1:
            output_per_decoder["Cat"] = {"Category":5 if tracking else 2}
        decoder_output_names = dict()
        for n, o_ns in output_per_decoder.items():
            decoder_output_names[n] = dict()
            for o_n, o_i in o_ns.items():
                decoder_output_names[n][o_n] = f"Output{o_i:02}_{o_n}"
        n_frame_pairs = n_frames -1
        if long_term:
            n_frame_pairs = n_frame_pairs + (frame_window-1) * (2 if next else 1)
        decoder_is_segmentation = {"Seg": True, "Center": True, "Track": False, "LinkMultiplicity": False} if tracking else {"Seg":True, "Center":True}
        if category_number > 1:
            decoder_is_segmentation["Cat"] = True
        skip_per_decoder = {"Seg": skip_connections, "Center": [], "Track": [], "LinkMultiplicity": [], "Cat":[]}

        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                for dSegName in output_per_decoder["Seg"].keys():
                    output_name = None if frame_window > 0 or predict_edm_derivatives else decoder_output_names["Seg"][dSegName]
                    decoder_out["Seg"][dSegName] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="tanh" if scale_edm else "linear", filters_out=1, l2_reg=l2_reg, layer_idx=l_idx, name=f"DecoderSeg{dSegName}", output_name=output_name)
                for dCenterName in output_per_decoder["Center"].keys():
                    output_name = None if frame_window > 0 or predict_cdm_derivatives else decoder_output_names["Center"][dCenterName]
                    decoder_out["Center"][dCenterName] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, l2_reg=l2_reg, layer_idx=l_idx, name=f"DecoderCenter{dCenterName}", output_name=output_name)
                if category_number > 1:
                    for dCatName in output_per_decoder["Cat"].keys():
                        output_name = None if frame_window > 0 else decoder_output_names["Cat"][dCatName]
                        decoder_out["Cat"][dCatName] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="softmax", filters_out=category_number, l2_reg=l2_reg, layer_idx=l_idx, name=f"Decoder{dCatName}", output_name=output_name)
                if tracking:
                    for dTrackName in output_per_decoder["Track"].keys():
                        decoder_out["Track"][dTrackName] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, l2_reg=l2_reg, layer_idx=l_idx, name=f"DecoderTrack{dTrackName}")
                    for dLinkMultiplicityName in output_per_decoder["LinkMultiplicity"].keys():
                        decoder_out["LinkMultiplicity"][dLinkMultiplicityName] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="softmax", filters_out=3, l2_reg=l2_reg, layer_idx=l_idx, name=f"Decoder{dLinkMultiplicityName}")
            else:
                for decoder_name, d_layers in decoder_layers.items():
                    d_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation="relu", l2_reg=l2_reg, layer_idx=l_idx, name=f"Decoder{decoder_name}") )

        # Create GRAPH
        if n_inputs == 1:
            inputs = [ tf.keras.layers.Input(shape=spatial_dimensions + [n_frames], name="Input") ]
            input_merged = ChannelToBatch(compensate_gradient=False, add_channel_axis=True,  name="MergeInputs")(inputs[0]) if frame_window > 0 else inputs[0]
        else:
            if frame_window > 0:
                inputs = [tf.keras.layers.Input(shape=spatial_dimensions + [n_frames], name=f"Input{i}") for i in range(n_inputs)]
                input_stacked = Stack(axis = -2, name="InputStack")(inputs)
                input_merged = ChannelToBatch(compensate_gradient=False, add_channel_axis=False, name="MergeInputs")(input_stacked)
            else:
                inputs = [tf.keras.layers.Input(shape=spatial_dimensions + [1], name=f"Input{i}") for i in range(n_inputs)]
                input_merged = tf.keras.layers.Concatenate(axis=-1, name="MergeInputs")(inputs)
        print(f"input dims: {n_inputs} x {spatial_dimensions} frames={n_frames}")

        downsampled = [input_merged]
        residuals = []
        for l in encoder_layers:
            down, res = l(downsampled[-1])
            downsampled.append(down)
            residuals.append(res)
        residuals = residuals[::-1]
        feature = downsampled[-1]
        for op in feature_convs:
            feature = op(feature)

        # combine individual features
        all_features = SplitBatch(n_frames, compensate_gradient = False, name = "SplitFeatures")(feature) if frame_window > 0 else [feature]
        combined_features = combine_features_op(all_features)

        # frame pairs
        if frame_window > 0 :
            feature_prev = []
            feature_next = []
            for i in range(1, n_frames):
                feature_prev.append(all_features[i-1])
                feature_next.append(all_features[i])
            if long_term:
                for c in range(0, frame_window-1):
                    feature_prev.append(all_features[c])
                    feature_next.append(all_features[frame_window])
                if next:
                    for c in range(frame_window+2, n_frames):
                        feature_prev.append(all_features[frame_window])
                        feature_next.append(all_features[c])
            feature_prev = tf.keras.layers.Concatenate(axis = 0, name="FeaturePairPrevToBatch")(feature_prev)
            feature_next = tf.keras.layers.Concatenate(axis = 0, name="FeaturePairNextToBatch")(feature_next)
            if attention>0:
                attention_result = attention_op([feature_prev, feature_next])
                feature_pair = pair_combine_op([feature_prev, feature_next, attention_result])
            else:
                feature_pair = pair_combine_op([feature_prev, feature_next])

            all_feature_pairs = SplitBatch(n_frame_pairs, compensate_gradient = False, name = "SplitFeaturePairs")(feature_pair)
            combined_feature_pairs = all_pair_combine_op(all_feature_pairs)
            combined_features = feature_pair_feature_combine_op([combined_features, combined_feature_pairs])
            for op in feature_blending_convs:
                combined_features = op(combined_features)

            feature_per_frame = NConvToBatch2D(compensate_gradient = True, n_conv = n_frames, inference_idx=frame_window, filters = feature_filters, name =f"SegmentationFeatures")(combined_features) # (N_CHAN x B, Y, X, F)
            feature_per_frame_pair = NConvToBatch2D(compensate_gradient = True, n_conv = n_frame_pairs, inference_idx=inference_pair_idx, filters = feature_filters, name =f"TrackingFeatures")(combined_features) # (N_PAIRS x B, Y, X, F)
        else:
            feature_per_frame = combined_features

        if len(encoder_settings) in skip_connections and frame_window > 0: # skip connection at feature level
            feature_skip = SelectFeature(inference_idx=frame_window, name ="SelectFeature")([feature, all_features])
            feature_per_frame = feature_skip_op([feature_skip, feature_per_frame])
            feature_pair_skip = SelectFeature(inference_idx=inference_pair_idx, name ="SelectFeaturePair")([feature_pair, all_feature_pairs])
            feature_per_frame_pair = feature_pair_skip_op([feature_pair_skip, feature_per_frame_pair])

        outputs=[]
        for decoder_name, is_segmentation in decoder_is_segmentation.items():
            if is_segmentation is not None:
                d_layers = decoder_layers[decoder_name]
                skip = skip_per_decoder[decoder_name]
                up = feature_per_frame if is_segmentation else feature_per_frame_pair
                for op in decoder_feature_op[decoder_name][0]:
                    up = op(up)
                for i, (l, res) in enumerate(zip(d_layers[::-1], residuals[:-1])):
                    up = l([up, res if len(d_layers) - i in skip else None])
                output_per_dec = dict()
                for output_name in output_per_decoder[decoder_name].keys():
                    if output_name in decoder_out[decoder_name]:
                        d_out = decoder_out[decoder_name][output_name]
                        layer_output_name = decoder_output_names[decoder_name][output_name]
                        if not is_segmentation and predict_next_displacement and not output_name.endswith(("FW", "BW")) or (decoder_name=="Seg" and predict_edm_derivatives or decoder_name== "Center" and predict_cdm_derivatives) and not output_name.endswith(("dX", "dY")):
                            layer_output_name += "_" # will be concatenated -> output name is used @ concat
                        fw = output_name.endswith("FW")
                        b2c_inference_idx = None if is_segmentation else (inference_pair_sel_fw if fw else inference_pair_sel_bw)
                        up_out = d_out([up, residuals[-1] if 0 in skip else None]) # (N_OUT x B, Y, X, F)
                        up_out = BatchToChannel(n_splits = n_frames if is_segmentation else n_frame_pairs, n_splits_inference= 1 if is_segmentation else len(inference_pair_idx), inference_idx=b2c_inference_idx, compensate_gradient = False, name = layer_output_name)(up_out) if frame_window > 0 else up_out
                        output_per_dec[output_name] = up_out
                if predict_next_displacement: # merge BW and FW outputs
                    for k in list(output_per_dec.keys()):
                        if k.endswith("FW"):
                            output_name_bw = k.replace("FW", "BW")
                            output_name = decoder_output_names[decoder_name][k].replace("FW", "")
                            output_per_dec[output_name] = tf.keras.layers.Concatenate(axis = -1, name = output_name)([output_per_dec.pop(output_name_bw), output_per_dec.pop(k)])
                if decoder_name=="Seg" and predict_edm_derivatives:
                    output_name = "EDM"
                    output_per_dec[output_name] = tf.keras.layers.Concatenate(axis=-1,  name=decoder_output_names[decoder_name][output_name])([output_per_dec[output_name], output_per_dec.pop("EDMdY"), output_per_dec.pop("EDMdX")])
                if decoder_name=="Center" and predict_cdm_derivatives:
                    output_name = "CDM"
                    output_per_dec[output_name] = tf.keras.layers.Concatenate(axis=-1,  name=decoder_output_names[decoder_name][output_name])([output_per_dec[output_name], output_per_dec.pop("CDMdY"), output_per_dec.pop("CDMdX")])
                outputs.extend(output_per_dec.values())
        return DiSTNetModel(inputs, outputs, name=name, frame_window=frame_window, next=next, spatial_dims=spatial_dimensions if attention > 0 or self_attention > 0 else None, long_term=long_term, predict_next_displacement=predict_next_displacement, predict_cdm_derivatives=predict_cdm_derivatives, predict_edm_derivatives=predict_edm_derivatives, category_number=category_number, **kwargs)

def encoder_op(param_list, downsampling_mode, skip_stop_gradient:bool = False, l2_reg:float=0, last_input_filters:int=0, attention_positional_encoding="2D", skip_parameters:tuple=None, name: str="EncoderLayer", layer_idx:int=1):
    name=f"{name}{layer_idx}"
    maxpool = downsampling_mode=="maxpool"
    maxpool_and_stride = downsampling_mode == "maxpool_and_stride"
    sequence, down_sequence, total_contraction, residual_filters, out_filters = parse_param_list(param_list, name, attention_positional_encoding=attention_positional_encoding, ignore_stride=maxpool, l2_reg=l2_reg, last_input_filters = last_input_filters if maxpool_and_stride else 0)
    assert total_contraction>1, "invalid parameters: no contraction specified"
    if maxpool:
        down_sequence = []
    if maxpool or maxpool_and_stride:
        down_sequence = down_sequence+[tf.keras.layers.MaxPool2D(pool_size=total_contraction, name=f"{name}_Maxpool{total_contraction}x{total_contraction}")]
        down_concat = tf.keras.layers.Concatenate(axis=-1, name = f"{name}_DownConcat", dtype="float32")
    def op(input):
        x = input
        if sequence is not None:
            for l in sequence:
                x=l(x)
        down = [l(x) for l in down_sequence]
        if len(down)>1:
            down = down_concat(down)
        else:
            down = down[0]
        if (sequence is not None or layer_idx>0) and skip_parameters is not None:
            res = x
            if skip_stop_gradient:
                res = stop_gradient(res, parent_name = name)
            n_splits, inference_idx = skip_parameters
            assert inference_idx<n_splits, f"invalid inference idx: {inference_idx} must be lower than n_splits: {n_splits}"
            feature_skip = SelectFeature(inference_idx=inference_idx,  name=f"{name}_SelectFeature")
            feature_split = SplitBatch(n_splits, compensate_gradient=False, name=f"{name}_SplitFeature")
            res = feature_skip([res, feature_split(res)])
        else:
            res = None
        return down, res
    return op, total_contraction, residual_filters, out_filters

def decoder_op(
            filters: int,
            filters_out: int = None,
            size_factor: int=2,
            conv_kernel_size:int=3,
            up_kernel_size:int=0,
            mode:str="tconv", # tconv, up_nn, up_bilinear,
            skip_combine_mode:str = "conv", # conv, sum, wsconv
            combine_kernel_size:int = 1,
            batch_norm:bool = False,
            weight_scaled:bool = False,
            dropout_rate:float=0,
            batch_norm_up:bool = False,
            weight_scaled_up:bool = False,
            dropout_rate_up:float=0,
            activation: str="relu",
            activation_out : str = None,
            op:str = "conv", # conv,resconv2d, resconv2d
            weighted_sum:bool=False, # in case op = resconv2d, resconv2d
            n_conv:int = 1,
            l2_reg:float=0,
            name: str="DecoderLayer",
            output_name: str = None,
            layer_idx:int=1,
        ):
        if layer_idx > 0:
            name=f"{name}{layer_idx}"
        if n_conv==0 and activation_out is not None:
            activation = activation_out
        if n_conv>0 and activation_out is None:
            activation_out = activation
        if n_conv>0 and filters_out is None:
            filters_out = filters

        up_op = upsampling_op(filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, weight_scaled = weight_scaled_up, batch_norm=batch_norm_up, dropout_rate=dropout_rate_up, l2_reg=l2_reg)
        up_op_out = upsampling_op(filters=filters_out, parent_name=None if output_name is not None else name, name = output_name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, weight_scaled = weight_scaled_up, batch_norm=batch_norm_up, dropout_rate=dropout_rate_up, l2_reg=l2_reg)
        if skip_combine_mode.lower()=="conv" or skip_combine_mode.lower()=="wsconv":
            combine = Combine(name = output_name if output_name is not None and n_conv==0 else name+"_combine", filters=filters if filters_out is None or n_conv>0 else filters_out, kernel_size = combine_kernel_size, l2_reg=l2_reg, weight_scaled=skip_combine_mode.lower()=="wsconv")
        else:
            combine = None
        op = op.lower().replace("_", "")
        if op == "res1d" or op=="resconv1d":
            raise NotImplementedError("ResConv1D are not implemented")
        elif op == "res2d" or op=="resconv2d":
            convs = [ResConv2D(kernel_size=conv_kernel_size, activation=activation_out if i==n_conv-1 else activation, weight_scaled=weight_scaled, batch_norm=batch_norm, dropout_rate=dropout_rate, l2_reg=l2_reg, weighted_sum=weighted_sum, name=f"{name}_ResConv2D{i}_{ker_size_to_string(conv_kernel_size)}") for i in range(n_conv)]
        else:
            if weight_scaled:
                convs = [WSConv2D(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, padding='same', activation=activation_out if i==n_conv-1 else activation, dropout_rate=dropout_rate, name=f"{name}_Conv{i}_{ker_size_to_string(conv_kernel_size)}" if i < n_conv-1 or output_name is None else output_name) for i in range(n_conv)]
            elif batch_norm or dropout_rate>0:
                convs = [Conv2DBNDrop(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, activation=activation_out if i==n_conv-1 else activation, batch_norm=batch_norm, dropout_rate=dropout_rate, l2_reg=l2_reg, name=f"{name}_Conv{i}_{ker_size_to_string(conv_kernel_size)}"if i < n_conv-1 or output_name is None else output_name) for i in range(n_conv)]
            else:
                convs = [tf.keras.layers.Conv2D(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, padding='same', activation=activation_out if i==n_conv-1 else activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg>0 else None, name=f"{name}_Conv{i}_{ker_size_to_string(conv_kernel_size)}" if i < n_conv-1 or output_name is None else output_name) for i in range(n_conv)]
        def op(input):
            down, res = input
            up = up_op_out(down) if n_conv==0 and res is None else up_op(down)
            if res is not None:
                if combine is not None:
                    up = combine([up, res])
                else:
                    up = up + res
            for c in convs:
                up = c(up)
            return up
        return op

def upsampling_op(
            filters: int,
            parent_name:str,
            size_factor:int=2,
            kernel_size: int=0,
            mode:str="tconv", # tconv, up_nn, up_bilinear
            norm_layer:str=None,
            activation: str="relu",
            batch_norm:bool = False,
            weight_scaled:bool = False,
            dropout_rate:float = 0,
            use_bias:bool = True,
            l2_reg:float = 0,
            name: str= None,
        ):
        assert mode in ["tconv", "up_nn", "up_bilinear"], "invalid mode"
        if kernel_size<size_factor:
            kernel_size = size_factor
        if mode=="tconv":
            if weight_scaled:
                raise NotImplementedError("Weight scaled transpose conv is not implemented")
            elif batch_norm or dropout_rate>0:
                upsample = Conv2DTransposeBNDrop(filters=filters, kernel_size=kernel_size, strides=size_factor, activation=activation, batch_norm=batch_norm, dropout_rate=dropout_rate, l2_reg=l2_reg, name=f"{parent_name}_tConv{ker_size_to_string(kernel_size)}" if parent_name is not None else name)
            else:
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg>0 else None
                upsample = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=size_factor, padding='same', activation=activation, use_bias=use_bias, kernel_regularizer=kernel_regularizer, name=f"{parent_name}_tConv{ker_size_to_string(kernel_size)}" if parent_name is not None else name)
            conv=None
        else:
            interpolation = "nearest" if mode=="up_nn" else 'bilinear'
            upsample = tf.keras.layers.UpSampling2D(size=size_factor, interpolation=interpolation, name = f"{parent_name}_Upsample{size_factor}x{size_factor}_{interpolation}" if parent_name is not None else name)
            if batch_norm:
                conv = Conv2DBNDrop(filters=filters, kernel_size=kernel_size, strides=1, batch_norm=batch_norm, dropout_rate=dropout_rate, l2_reg=l2_reg, name=f"{parent_name}_Conv{ker_size_to_string(kernel_size)}" if parent_name is not None else name, activation=activation )
            else:
                conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', name=f"{parent_name}_Conv{ker_size_to_string(kernel_size)}" if parent_name is not None else name, use_bias=use_bias, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg>0 else None )
        def op(input):
            x = upsample(input)
            if conv is not None:
                x = conv(x)
            return x
        return op

def stop_gradient(input, parent_name:str, name:str="StopGradient"):
    if parent_name is not None and len(parent_name)>0:
        name = f"{parent_name}_{name}"
    sg = StopGradient(name=name)
    return sg(input)

def parse_param_list(param_list, name:str, last_input_filters:int=0, ignore_stride:bool = False, attention_positional_encoding="2D", l2_reg:float=0):
    if param_list is None or len(param_list)==0:
        return [], None, 1, 0, 0
    total_contraction = 1
    if ignore_stride:
        param_list = [params.copy() for params in param_list]
        for params in param_list:
            total_contraction *= params.get("downscale", 1)
            params["downscale"] = 1
    # split into sequence with no stride (for residual) and the rest of the sequence
    i = 0
    if param_list[0].get("downscale", 1)==1:
        residual_filters = last_input_filters
        sequence = []
        while i<len(param_list) and param_list[i].get("downscale", 1) == 1:
            if "filters" in param_list[i]:
                if isinstance(param_list[i]["filters"], float):
                    assert residual_filters>0, "last_input_filters should be >0 when filters is a float"
                    param_list[i]["filters"] = int(param_list[i]["filters"] * residual_filters+0.5)
                residual_filters =  param_list[i]["filters"]
            if "l2_reg" not in param_list[i]:
                param_list[i]["l2_reg"] = l2_reg
            sequence.append(parse_params(**param_list[i], attention_positional_encoding=attention_positional_encoding, name = f"{name}_Op{i}"))
            i+=1
    else:
        sequence=None
        residual_filters = 0
    if i<len(param_list):
        if i==len(param_list)-1:
            params = param_list[i].copy()
            filters = params.pop("filters")
            if "l2_reg" not in param_list[i]:
                param_list[i]["l2_reg"] = l2_reg
            out_filters = filters
            if last_input_filters>0: # case of stride + maxpool -> out filters -= input filters
                if residual_filters>0:
                    last_input_filters=residual_filters # input of downscaler is the residual
                filters -= last_input_filters
            down = [parse_params(**params, filters=filters, attention_positional_encoding=attention_positional_encoding, name=f"{name}_DownOp")] if filters > 0 else []
            total_contraction *= param_list[i].get("downscale", 1)
        else:
            raise ValueError("Only one downscale operation allowed")
    else:
        down = None
        out_filters = residual_filters
    return sequence, down, total_contraction, residual_filters, out_filters

def parse_params(filters:int = 0, kernel_size:int = 3, op:str = "conv", dilation:int=1, activation="relu", downscale:int=1, attention_positional_encoding:str="2D", dropout_rate:float=0, weight_scaled:bool=False, batch_norm:bool=False, weighted_sum:bool=False, l2_reg:float=0, split_conv:bool = False, num_attention_heads:int=1, name:str=""):
    op = op.lower().replace("_", "")
    if op =="res1d" or op=="resconv1d":
        raise NotImplementedError("ResConv1D is not implmeneted")
    elif op =="res2d" or op == "resconv2d":
        return ResConv2D(kernel_size=kernel_size, dilation=dilation, activation=activation, dropout_rate=dropout_rate, weight_scaled=weight_scaled, batch_norm=batch_norm, weighted_sum=weighted_sum, l2_reg=l2_reg, split_conv=split_conv, name=f"{name}_ResConv2D{ker_size_to_string(kernel_size)}")
    assert filters > 0 , "filters must be > 0"
    if op=="selfattention" or op=="sa":
        self_attention_op = SpatialAttention2D(num_heads=num_attention_heads, positional_encoding=attention_positional_encoding, dropout=dropout_rate, l2_reg=l2_reg, name=f"{name}_SelfAttention")
        self_attention_skip_op = Combine(filters=filters, l2_reg=l2_reg, name=f"{name}_SelfAttentionSkip")
        def op(x):
            sa = self_attention_op([x, x])
            return self_attention_skip_op([x, sa])
        return op
    if weight_scaled: # no l2_reg
        return WSConv2D(filters=filters, kernel_size=kernel_size, strides = downscale, dilation_rate = dilation, activation=activation, dropout_rate=dropout_rate, padding='same', name=f"{name}_Conv{ker_size_to_string(kernel_size)}")
    elif batch_norm or dropout_rate>0:
        return Conv2DBNDrop(filters=filters, kernel_size=kernel_size, strides = downscale, dilation = dilation, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm, l2_reg=l2_reg, name=f"{name}_Conv{ker_size_to_string(kernel_size)}")
    else:
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg) if l2_reg>0 else None
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides = downscale, dilation_rate = dilation, padding='same', activation=activation, kernel_regularizer=kernel_regularizer, name=f"{name}_Conv{ker_size_to_string(kernel_size)}")
