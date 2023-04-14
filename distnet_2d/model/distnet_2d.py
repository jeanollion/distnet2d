# gradient accumulation code from : # from https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/accumulators.py

import tensorflow as tf
import tensorflow_probability as tfp
from .layers import ConvNormAct, Bneck, UpSamplingLayer2D, StopGradient, Combine, WeigthedGradient, ResConv1D, ResConv2D, Conv2DBNDrop, Conv2DTransposeBNDrop, WSConv2D, WSConv2DTranspose, BatchToChannel2D, SplitBatch2D, ChannelToBatch2D, NConvToBatch2D
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from .attention import SpatialAttention2D
from ..utils.helpers import ensure_multiplicity, flatten_list
from .utils import get_layer_dtype
from ..utils.losses import weighted_binary_crossentropy, weighted_loss_by_category, balanced_category_loss, edm_contour_loss, balanced_background_binary_crossentropy, MeanSquaredErrorChannel, l2, get_grad_weight_fun
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from ..utils.lovasz_loss import lovasz_hinge
from ..utils.objectwise_motion_losses import get_motion_losses
from ..utils.agc import adaptive_clip_grad
from .gradient_accumulator import GradientAccumulator

class DistnetModel(Model):
    def __init__(self, *args, spatial_dims,
        edm_loss_weight:float=1, edm_lovasz_loss_weight:float=1, edm_lovasz_label_loss_weight:float=0,
        contour_loss_weight:float = 1,
        center_loss_weight:float=1, center_lovasz_loss_weight:float=1, center_unicity_loss_weight:float=1e-1,
        displacement_loss_weight:float=1, displacement_grad_weight:float=1, displacement_lovasz_loss_weight:float=0,
        center_displacement_loss_weight:float=1e-1, center_displacement_grad_weight_center:float=1e-1, center_displacement_grad_weight_displacement:float=1e-1, # ratio : init: center/motion = 10-100 . trained : motion/center = 10-100
        category_loss_weight:float=1,
        max_objects_number:int = 0,
        center_scale:float=0, # 0 : computed automatically
        edm_loss= MeanSquaredError(),
        center_loss = MeanSquaredError(),
        displacement_loss = MeanSquaredError(),
        category_weights = None, # array of weights: [background, normal, division, no previous cell] or None = auto
        category_class_frequency_range=[1/50, 50],
        category_background = True,
        next = True,
        frame_window = 1,
        predict_contours = False,
        predict_center = False,
        long_term:bool = False,
        gradient_safe_mode = False,
        gradient_log_dir:str=None,
        accum_steps=1, use_agc=False, agc_clip_factor=2, agc_eps=1e-3, agc_exclude_output=False,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.displacement_lovasz_weight = displacement_lovasz_loss_weight
        self.edm_weight = edm_loss_weight
        self.edm_lovasz_weight = edm_lovasz_loss_weight
        self.edm_lovasz_label_weight = edm_lovasz_label_loss_weight
        self.contour_weight = contour_loss_weight
        self.center_lovasz_weight = center_lovasz_loss_weight
        self.center_weight = center_loss_weight
        self.center_unicity_weight = center_unicity_loss_weight
        self.displacement_weight = displacement_loss_weight
        self.displacement_grad_weight = displacement_grad_weight
        self.center_displacement_weight = center_displacement_loss_weight
        self.category_weight = category_loss_weight
        self.gradient_safe_mode=gradient_safe_mode
        self.predict_contours = predict_contours
        self.predict_center = predict_center
        self.spatial_dims = spatial_dims
        self.next = next
        self.frame_window = frame_window
        self.edm_loss = edm_loss
        self.center_loss=center_loss
        self.contour_loss = MeanSquaredError()
        self.displacement_loss = displacement_loss
        self.motion_losses = get_motion_losses(spatial_dims, motion_range = frame_window * (2 if next else 1), center_displacement_grad_weight_center=center_displacement_grad_weight_center, center_displacement_grad_weight_displacement=center_displacement_grad_weight_displacement, center_scale=center_scale, next = next, frame_window=frame_window, long_term=long_term, center_motion = center_displacement_loss_weight>0, center_unicity=center_unicity_loss_weight>0, max_objects_number=max_objects_number)
        min_class_frequency=category_class_frequency_range[0]
        max_class_frequency=category_class_frequency_range[1]
        if category_weights is not None:
            if category_background:
                assert len(category_weights)==4, "4 category weights should be provided: background, normal cell, dividing cell, cell with no previous cell"
            else:
                assert len(category_weights)==3, "3 category weights should be provided: normal cell, dividing cell, cell with no previous cell"
            self.category_loss=weighted_loss_by_category(CategoricalCrossentropy(), category_weights, remove_background=not category_background)
        else:
            self.category_loss = balanced_category_loss(CategoricalCrossentropy(), 4 if category_background else 3, min_class_frequency=min_class_frequency, max_class_frequency=max_class_frequency, remove_background=not category_background)
        self.category_background = category_background
        # gradient accumulation from https://github.com/andreped/GradientAccumulator/blob/main/gradient_accumulator/accumulators.py
        self.long_term = long_term
        self.use_grad_acc = accum_steps>1
        self.accum_steps = float(accum_steps)
        if self.use_grad_acc or use_agc:
            self.gradient_accumulator = GradientAccumulator(accum_steps, self)
        self.use_agc = use_agc
        self.agc_clip_factor = agc_clip_factor
        self.agc_eps = agc_eps
        self.agc_exclude_keywords=["DecoderTrackY0/, DecoderTrackX0/", "DecoderCat0/", "DecoderCenter0/", "DecoderSegEDM0"] if agc_exclude_output else None
        if gradient_log_dir is not None:
            self.grad_writer = tf.summary.create_file_writer(gradient_log_dir)
        else:
            self.grad_writer = None
        self.loss_groups = [["edm", "edm_lh", "edm_lh_label", "contour", "center", "center_lh", "displacement", "displacement_lh", "category"], ["center_displacement", "center_unicity"]]

    def train_step(self, data):
        if self.use_grad_acc:
            self.gradient_accumulator.init_train_step()

        fw = self.frame_window
        n_frame_pairs = fw * (2 if self.next else 1)
        if self.long_term:
            n_frame_pairs += (fw - 1) * (2 if self.next else 1)
        mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        x, y = data
        displacement_weight = self.displacement_weight #/ 2. # y & x
        # displacement_weight /= (fw * (2. if self.next else 1)) # mean per channel # should it be divided by channel ?
        category_weight = self.category_weight / float(n_frame_pairs)
        contour_weight = self.contour_weight
        edm_weight = self.edm_weight
        center_weight = self.center_weight
        center_displacement_weight = self.center_displacement_weight
        inc = 1 if self.predict_contours else 0
        inc += 1 if self.predict_center else 0
        if len(y) == 7 + inc: # y = edm, contour, center, dy, dX, cat, true_center, prev_labels, label_rank
            labels, prev_labels, centers = y[-1], y[-2], y[-3]
        else :
            labels = None
            if self.predict_center and self.predict_contours:
                assert len(y) == 6, f"invalid number of output. Expected: 6 actual {len(y)}"
            elif self.predict_center or self.predict_contours:
                assert len(y) == 5, f"invalid number of output. Expected: 5 actual {len(y)}"
            else:
                assert len(y) == 4, f"invalid number of output. Expected: 4 actual {len(y)}"

        with tf.GradientTape(persistent=(self.use_agc and len(self.loss_groups)>1) or self.grad_writer is not None) as tape:
            y_pred = self(x, training=True)  # Forward pass
            # compute loss
            losses = dict()
            loss_weights = dict()

            inc=0
            if edm_weight>0:
                edm_loss = self.edm_loss(y[inc], y_pred[inc])#, sample_weight = weight_map)
                losses["edm"] = edm_loss
                loss_weights["edm"] = edm_weight
            if self.edm_lovasz_weight>0:
                edm_loss_lh = lovasz_hinge(y_pred[inc], tf.math.greater(y[inc], 0), channel_axis=True)
                losses["edm_lh"] = edm_loss_lh
                loss_weights["edm_lh"] = self.edm_lovasz_weight
            if self.edm_lovasz_label_weight>0 and labels is not None:
                edm_score = 2. * tf.math.exp(-tf.math.square(y[0]-y_pred[0])) - 1.
                edm_loss_lh = lovasz_hinge(edm_score, labels, per_label=True, channel_axis=True)
                losses["edm_lh_label"] = edm_loss_lh
                loss_weights["edm_lh_label"] = self.edm_lovasz_label_weight

            if self.predict_contours:
                inc+=1
                contour_loss = self.contour_loss(y[inc], y_pred[inc])
                losses["contour"] = contour_loss
                loss_weights["contour"] = contour_weight

            if self.predict_center:
                inc+=1
                if center_weight>0:
                    center_pred_inside=tf.where(tf.math.greater(y[0], 0), y_pred[inc], 0) # do not predict anything outside
                    center_loss = self.center_loss(y[inc], center_pred_inside)
                    losses["center"] = center_loss
                    loss_weights["center"] = center_weight

                if self.center_lovasz_weight>0 and labels is not None:
                    # @tf.custom_gradient
                    # def p_pred(x):
                    #     def grad(dy):
                    #         p = tf.boolean_mask(tf.math.abs(dy), tf.math.not_equal(dy, 0.))
                    #         print(f"pred grad: \n{tf.math.reduce_mean(p)}")
                    #         return dy
                    #     return x, grad
                    # pred = p_pred(y_pred[inc])
                    score = 2. * tf.math.exp(-tf.math.square(y_pred[inc] - y[inc])) - 1.
                    #@tf.custom_gradient
                    # def p_score(x):
                    #     def grad(dy):
                    #         p = tf.boolean_mask(tf.math.abs(dy), tf.math.not_equal(dy, 0.))
                    #         print(f"score grad: \n{tf.math.reduce_mean(p)}")
                    #         return dy
                    #     return x, grad
                    # score = p_score(score)
                    center_loss_lh = lovasz_hinge(score, labels, per_label=True, channel_axis=True)
                    losses["center_lh"] = center_loss_lh
                    loss_weights["center_lh"] = self.center_lovasz_weight

            if self.displacement_lovasz_weight>0:
                score_y = 2. * tf.math.exp(-tf.math.square(y[1+inc]-y_pred[1+inc])) - 1.
                score_x = 2. * tf.math.exp(-tf.math.square(y[2+inc]-y_pred[2+inc])) - 1.
                d_loss = lovasz_hinge(score_y, labels[...,1:], per_label=True, channel_axis=True) + lovasz_hinge(score_x, labels[...,1:], per_label=True, channel_axis=True)
                losses["displacement_lh"] = d_loss
                loss_weights["displacement_lh"] = self.displacement_lovasz_weight

            #regression displacement loss
            if displacement_weight>0:
                mask = tf.math.greater(y[0][...,1:], 0)
                if self.long_term and fw>1:
                    mask_lt = tf.tile(mask[...,fw-1:fw], [1, 1, 1, fw-1])
                    if self.next:
                        mask = tf.concat([mask, mask_lt, mask[...,fw+1:]], -1)
                    else:
                        mask = tf.concat([mask, mask_lt], -1)
                dy_inside=tf.where(mask, y_pred[1+inc], 0) # do not predict anything outside
                dx_inside=tf.where(mask, y_pred[2+inc], 0) # do not predict anything outside
                dy_norm = _get_abs_mean_foreground(y[1+inc], mask)
                dx_norm = _get_abs_mean_foreground(y[2+inc], mask)
                #print(f"dx norm: {dx_norm} dy norm: {dy_norm}")
                if self.displacement_grad_weight!=1:
                    g_weight = get_grad_weight_fun(self.displacement_grad_weight)
                    dy_inside = g_weight(dy_inside)
                    dx_inside = g_weight(dx_inside)
                d_loss = self.displacement_loss(y[1+inc], dy_inside)/dy_norm + self.displacement_loss(y[2+inc], dx_inside)/dx_norm
                losses["displacement"] = d_loss
                loss_weights["displacement"] = displacement_weight

            n_motion_loss =(1 if center_displacement_weight>0 else 0) + (1 if self.center_unicity_weight>0 else 0)
            if n_motion_loss>0:
                motion_losses = self.motion_losses(y_pred[1+inc], y_pred[2+inc], y_pred[inc], labels, prev_labels, centers)
                if center_displacement_weight>0:
                    center_motion_loss = motion_losses[0] if n_motion_loss>1 else motion_losses
                    losses["center_displacement"] = center_motion_loss
                    loss_weights["center_displacement"] = center_displacement_weight
                    #center_motion_loss_norm = tf.math.divide_no_nan(center_displacement_weight, tf.stop_gradient(center_motion_loss))
                    #loss = loss + center_motion_loss * center_displacement_weight
                if self.center_unicity_weight>0:
                    center_unicity_loss = motion_losses[1] if n_motion_loss>1 else motion_losses
                    losses["center_unicity"] = center_unicity_loss
                    loss_weights["center_unicity"] = self.center_unicity_weight

            # category loss
            cat_loss = 0
            for i in range(n_frame_pairs):
                if not self.category_background:
                    inside_mask = tf.math.greater(y[3+inc][...,i:i+1], 0)
                    cat_pred_inside=tf.where(inside_mask, y_pred[3+inc][...,3*i:3*i+3], 1)
                    cat_loss = cat_loss + self.category_loss(y[3+inc][...,i:i+1], cat_pred_inside)
                else:
                    cat_loss = cat_loss + self.category_loss(y[3+inc][...,i:i+1], y_pred[3+inc][...,4*i:4*i+4])
            losses["category"] = cat_loss
            loss_weights["category"] = category_weight

            losses["loss"] = 0.
            loss_per_group = []
            for g in self.loss_groups:
                loss = 0.
                loss_count = 0
                for n in g:
                    if n in losses:
                        loss = loss + losses[n] * loss_weights[n]
                        loss_count = loss_count + 1
                if loss_count>0:
                    losses["loss"] = losses["loss"] + loss
                    if mixed_precision:
                        loss = self.optimizer.get_scaled_loss(loss)
                    if self.use_agc or len(loss_per_group)==0:
                        loss_per_group.append(loss)
                    else:
                        loss_per_group[0] = loss_per_group[0] + loss # no need to have separate losses

        if self.grad_writer is not None:
            trainable_vars_tape = [t for t in self.trainable_variables if (t.name.startswith("DecoderSegEDM") or t.name.startswith("DecoderCenter0") or t.name.startswith("DecoderTrackY") or t.name.startswith("DecoderCat0") or t.name.startswith("FeatureSequence/Op4") or t.name.startswith("Attention")) and ("/kernel" in t.name or "/wv" in t.name) ]
            with self.grad_writer.as_default(step=self._train_counter):
                for loss_name, loss_value in losses.items():
                    if loss_name != "loss" :
                        w = loss_weights[loss_name]
                        gradients = tape.gradient(loss_value, trainable_vars_tape)
                        if mixed_precision:
                            gradients = self.optimizer.get_unscaled_gradients(grad)
                        for v, g in zip(trainable_vars_tape, gradients):
                            if g is not None:
                                g = g * w
                                #tf.summary.histogram(f"grad_{v.name}_loss-{loss_name}", g)
                                print(f"{v.name}, loss: {loss_name}, val: {loss_value}, grad: [{tf.math.reduce_min(g).numpy()}, {tf.reduce_mean(g).numpy()} {tf.reduce_mean(tf.math.abs(g)).numpy()}, {tf.math.reduce_max(g).numpy()}] shape: {g.shape}")
                        if self.use_agc:
                            gradients = adaptive_clip_grad(trainable_vars_tape, gradients, clip_factor=self.agc_clip_factor, eps=self.agc_eps, exclude_keywords=self.agc_exclude_keywords, grad_scale = w)
                            for v, g in zip(trainable_vars_tape, gradients):
                                if g is not None:
                                    #tf.summary.histogram(f"grad_{v.name}_loss-{loss_name}", g)
                                    print(f"AGC: layer: {v.name}, loss: {loss_name}, grad: [{tf.math.reduce_min(g).numpy()}, {tf.reduce_mean(g).numpy()} {tf.reduce_mean(tf.math.abs(g)).numpy()}, {tf.math.reduce_max(g).numpy()}]")

        # Compute gradients
        for loss in loss_per_group:
            gradients = tape.gradient(loss, self.trainable_variables)
            if mixed_precision:
                gradients = self.optimizer.get_unscaled_gradients(gradients)
            if self.use_agc:
                gradients = adaptive_clip_grad(self.trainable_variables, gradients, clip_factor=self.agc_clip_factor, eps=self.agc_eps, exclude_keywords=self.agc_exclude_keywords)
            if not self.use_grad_acc and len(loss_per_group)==1:
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables)) #Update weights
            else:
                self.gradient_accumulator.accumulate_gradients(gradients)
        if self.use_grad_acc or len(loss_per_group)>1:
            self.gradient_accumulator.apply_gradients()
        return losses
        # gradient safe mode
        # def t_fn():# if any of the gradients are NaN, set loss metric to NaN
        #     return losses["loss"]
        #     #return tf.constant(float('NaN'))
        # def f_fn(): # if all gradients are valid apply them
        #     self.optimizer.apply_gradients(grads_and_vars, experimental_aggregate_gradients=False)
        #     return losses["loss"]
        # grad_nan=[tf.reduce_any(tf.math.is_nan(g)) for g,_ in grads_and_vars if g is not None]
        # grad_nan=tf.reduce_any(grad_nan)
        # losses["loss"] = tf.cond(grad_nan, t_fn, f_fn)
        # # printlosses={k:v.numpy() for k,v in losses.items()}
        # # print(f"losses: {printlosses}")
        # return losses

    def set_inference(self, inference:bool=True):
        for layer in self.layers:
            if isinstance(layer, (NConvToBatch2D, BatchToChannel2D)):
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

def _get_abs_mean_foreground(data, mask):
    data = tf.reshape(tf.math.abs(data), [-1])
    mask = tf.reshape(mask, [-1])
    data = tf.boolean_mask(data, mask)
    return tf.cond(tf.equal(tf.shape(data)[0], 0),
                   lambda: 1.,
                   lambda: tf.reduce_mean(data))

def get_distnet_2d(input_shape,
            frame_window:int,
            next:bool,
            config,
            extract_per_decoder_type:bool = False,
            name: str="DiSTNet2D",
            **kwargs):
    fun = get_distnet_2d_erf2 if extract_per_decoder_type else get_distnet_2d_erf
    return fun(input_shape, upsampling_mode = config.upsampling_mode, downsampling_mode=config.downsampling_mode, combine_kernel_size=config.combine_kernel_size, skip_stop_gradient=False, skip_connections=False, encoder_settings=config.encoder_settings, feature_settings=config.feature_settings, feature_blending_settings=config.feature_blending_settings, decoder_settings=config.decoder_settings, feature_decoder_settings=config.feature_decoder_settings, attention=True, frame_window=frame_window, next=next, name=name, **kwargs)

def get_distnet_2d_erf(input_shape, # Y, X
            encoder_settings:list,
            feature_settings: list,
            feature_blending_settings: list,
            feature_decoder_settings:list,
            decoder_settings: list,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "maxpool_and_stride", #maxpool, stride, maxpool_and_stride
            combine_kernel_size:int = 1,
            skip_stop_gradient:bool = True,
            skip_connections:bool = False,
            skip_combine_mode:str="conv", #conv, wsconv
            attention : bool = True,
            frame_window:int = 1,
            next:bool=True,
            long_term:bool = False,
            predict_center = True,
            category_background = True,
            name: str="DiSTNet2D",
            **kwargs,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        spatial_dims = ensure_multiplicity(2, input_shape)
        if isinstance(spatial_dims, tuple):
            spatial_dims = list(spatial_dims)
        if frame_window<=1:
            long_term = False
        n_chan = frame_window * (2 if next else 1) + 1
        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        no_residual_layer = []
        last_input_filters = 1
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters, out_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, last_input_filters = last_input_filters, layer_idx = l_idx)
            last_input_filters = out_filters
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            no_residual_layer.append(residual_filters==0)
        # define feature operations
        feature_convs, _, _, feature_filters, _ = parse_param_list(feature_settings, "FeatureSequence", last_input_filters=out_filters)
        combine_filters = int(feature_filters * n_chan / 2.)
        combine_features_op = Combine(filters=combine_filters, compensate_gradient = True, name="CombineFeatures")
        if attention:
            attention_op = SpatialAttention2D(positional_encoding="2D", name="Attention")
            attention_combine = Combine(filters=combine_filters, compensate_gradient = True, name="AttentionCombine")
            attention_skip_op = Combine(filters=combine_filters, name="AttentionSkip")
        for f in feature_blending_settings:
            if "filters" not in f or f["filters"]<0:
                f["filters"] = combine_filters
        feature_blending_convs, _, _, feature_blending_filters, _ = parse_param_list(feature_blending_settings, "FeatureBlendingSequence", last_input_filters=combine_filters)

        # define decoder operations
        decoder_layers={"Seg":[], "Center":[], "Track":[], "Cat":[]}
        get_seq_and_filters = lambda l : [l[i] for i in [0, 3]]
        decoder_feature_op={n: get_seq_and_filters(parse_param_list(feature_decoder_settings, f"FeatureBlending{n}", last_input_filters=feature_blending_filters)) for n in decoder_layers.keys()}
        decoder_out={"Seg":{}, "Center":{}, "Track":{}, "Cat":{}}
        output_per_decoder = {"Seg": ["EDM"], "Center": ["Center"], "Track": ["dY", "dX"], "Cat": ["Cat"]}
        n_frame_pairs = n_chan -1
        if long_term:
            n_frame_pairs = n_frame_pairs + (frame_window-1) * (2 if next else 1)
        n_output_per_decoder = {"Seg": [n_chan, frame_window], "Center": [n_chan, frame_window] if predict_center else [0, 0], "Track": [n_frame_pairs, frame_window-1], "Cat": [n_frame_pairs, frame_window-1]}
        skip_per_decoder = {"Seg": skip_connections, "Center": False, "Track": False, "Cat": False}
        output_inc = 0
        seg_out = ["Output0_EDM"]
        activation_out = ["linear"]

        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out["Seg"]["EDM"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderSegEDM")
                decoder_out["Track"]["dY"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderTrackY")
                decoder_out["Track"]["dX"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderTrackX")
                decoder_out["Cat"]["Cat"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="softmax", filters_out=4 if category_background else 3, layer_idx=l_idx, name=f"DecoderCat")
                if predict_center:
                    decoder_out["Center"]["Center"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderCenter")
            else:
                for decoder_name, d_layers in decoder_layers.items():
                    d_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx, name=f"Decoder{decoder_name}") )
        decoder_output_names = dict()
        oi = 0
        for n, o_ns in output_per_decoder.items():
            decoder_output_names[n] = dict()
            for o_n in o_ns:
                decoder_output_names[n][o_n] = f"Output{oi}_{o_n}"
                oi += 1
        # Create GRAPH
        input = tf.keras.layers.Input(shape=spatial_dims+[n_chan], name="Input")
        input_merged = ChannelToBatch2D(compensate_gradient = False, name = "MergeInputs")(input)
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

        all_features = SplitBatch2D(n_chan, compensate_gradient = False, name = "SplitFeatures")(feature)
        combined_features = combine_features_op(all_features)
        if attention:
            attention_result = []
            for i in range(1, n_chan):
                attention_result.append(attention_op([all_features[i-1], all_features[i]]))
            if long_term:
                for c in range(0, frame_window-1):
                    attention_result.append(attention_op([all_features[c], all_features[frame_window]]))
                if next:
                    for c in range(frame_window+2, n_chan):
                        attention_result.append(attention_op([all_features[frame_window], all_features[c]]))
            attention = attention_combine(attention_result)
            combined_features = attention_skip_op([attention, combined_features])

        for op in feature_blending_convs:
            combined_features = op(combined_features)

        outputs = []
        for decoder_name, [n_out, out_idx] in n_output_per_decoder.items():
            if n_out>0:
                d_layers = decoder_layers[decoder_name]
                for output_name in output_per_decoder[decoder_name]:
                    if output_name in decoder_out[decoder_name]:
                        d_out = decoder_out[decoder_name][output_name]
                        layer_output_name = decoder_output_names[decoder_name][output_name]
                        skip = skip_per_decoder[decoder_name]
                        decoder_features = combined_features
                        for op in decoder_feature_op[decoder_name][0]:
                            decoder_features = op(decoder_features)
                        up = NConvToBatch2D(compensate_gradient = True, n_conv = n_out, inference_conv_idx=out_idx, filters = min(decoder_feature_op[decoder_name][1], feature_filters//2), name = f"FeatureConv{decoder_name}{output_name}")(decoder_features) # (N_OUT x B, Y, X, F)
                        for l, res in zip(d_layers[::-1], residuals[:-1]):
                            up = l([up, res if skip else None])
                        up = d_out([up, residuals[-1] if skip else None]) # (N_OUT x B, Y, X, F)
                        up = BatchToChannel2D(n_splits = n_out, compensate_gradient = False, name = layer_output_name)(up)
                        outputs.append(up)
        return DistnetModel([input], outputs, name=name, frame_window=frame_window, next = next, predict_contours = False, predict_center=predict_center, spatial_dims=spatial_dims, long_term=long_term, category_background=category_background, **kwargs)

def get_distnet_2d_erf2(input_shape, # Y, X
            encoder_settings:list,
            feature_settings: list,
            feature_blending_settings: list,
            feature_decoder_settings:list,
            decoder_settings: list,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "maxpool_and_stride", #maxpool, stride, maxpool_and_stride
            combine_kernel_size:int = 1,
            skip_stop_gradient:bool = True,
            skip_connections:bool = False,
            skip_combine_mode:str="conv", #conv, wsconv
            attention : bool = True,
            frame_window:int = 1,
            next:bool=True,
            long_term:bool = True,
            predict_center = True,
            category_background = False,
            name: str="DiSTNet2D",
            **kwargs,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        spatial_dims = ensure_multiplicity(2, input_shape)
        if isinstance(spatial_dims, tuple):
            spatial_dims = list(spatial_dims)
        if frame_window<=1:
            long_term = False
        n_chan = frame_window * (2 if next else 1) + 1
        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        no_residual_layer = []
        last_input_filters = 1
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters, out_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, last_input_filters = last_input_filters, layer_idx = l_idx)
            last_input_filters = out_filters
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            no_residual_layer.append(residual_filters==0)
        # define feature operations
        feature_convs, _, _, feature_filters, _ = parse_param_list(feature_settings, "FeatureSequence", last_input_filters=out_filters)
        combine_filters = int(feature_filters * n_chan / 2.)
        combine_features_op = Combine(filters=combine_filters, compensate_gradient = True, name="CombineFeatures")
        if attention:
            attention_op = SpatialAttention2D(positional_encoding="2D", name="Attention")
            attention_combine = Combine(filters=combine_filters, compensate_gradient = True, name="AttentionCombine")
            attention_skip_op = Combine(filters=combine_filters, name="AttentionSkip")
        for f in feature_blending_settings:
            if "filters" not in f or f["filters"]<0:
                f["filters"] = combine_filters
        feature_blending_convs, _, _, feature_blending_filters, _ = parse_param_list(feature_blending_settings, "FeatureBlendingSequence", last_input_filters=combine_filters)

        # define decoder operations
        decoder_layers={"Seg":[], "Center":[], "Track":[], "Cat":[]}
        get_seq_and_filters = lambda l : [l[i] for i in [0, 3]]
        decoder_feature_op={n: get_seq_and_filters(parse_param_list(feature_decoder_settings, f"Features{n}", last_input_filters=feature_filters)) for n in decoder_layers.keys()}
        decoder_out={"Seg":{}, "Center":{}, "Track":{}, "Cat":{}}
        output_per_decoder = {"Seg": ["EDM"], "Center": ["Center"], "Track": ["dY", "dX"], "Cat": ["Cat"]}
        n_frame_pairs = n_chan -1
        if long_term:
            n_frame_pairs = n_frame_pairs + (frame_window-1) * (2 if next else 1)
        decoder_is_segmentation = {"Seg": True, "Center": True, "Track": False, "Cat": False}
        skip_per_decoder = {"Seg": skip_connections, "Center": False, "Track": False, "Cat": False}
        output_inc = 0
        seg_out = ["Output0_EDM"]
        activation_out = ["linear"]

        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out["Seg"]["EDM"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderSegEDM")
                decoder_out["Track"]["dY"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderTrackY")
                decoder_out["Track"]["dX"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderTrackX")
                decoder_out["Cat"]["Cat"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="softmax", filters_out=4 if category_background else 3, layer_idx=l_idx, name=f"DecoderCat")
                if predict_center:
                    decoder_out["Center"]["Center"] = decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation_out="linear", filters_out=1, layer_idx=l_idx, name=f"DecoderCenter")
            else:
                for decoder_name, d_layers in decoder_layers.items():
                    d_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], mode=upsampling_mode, skip_combine_mode=skip_combine_mode, combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx, name=f"Decoder{decoder_name}") )
        decoder_output_names = dict()
        oi = 0
        for n, o_ns in output_per_decoder.items():
            decoder_output_names[n] = dict()
            for o_n in o_ns:
                decoder_output_names[n][o_n] = f"Output{oi}_{o_n}"
                oi += 1
        # Create GRAPH
        input = tf.keras.layers.Input(shape=spatial_dims+[n_chan], name="Input")
        input_merged = ChannelToBatch2D(compensate_gradient = False, name = "MergeInputs")(input)
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

        all_features = SplitBatch2D(n_chan, compensate_gradient = False, name = "SplitFeatures")(feature)
        combined_features = combine_features_op(all_features)
        if attention:
            attention_result = []
            for i in range(1, n_chan):
                attention_result.append(attention_op([all_features[i-1], all_features[i]]))
            if long_term:
                for c in range(0, frame_window-1):
                    attention_result.append(attention_op([all_features[c], all_features[frame_window]]))
                if next:
                    for c in range(frame_window+2, n_chan):
                        attention_result.append(attention_op([all_features[frame_window], all_features[c]]))
            attention = attention_combine(attention_result)
            combined_features = attention_skip_op([attention, combined_features])

        for op in feature_blending_convs:
            combined_features = op(combined_features)

        outputs = []
        feature_per_frame = NConvToBatch2D(compensate_gradient = True, n_conv = n_chan, inference_conv_idx=frame_window, filters = feature_filters, name = f"SegmentationFeatures")(combined_features) # (N_OUT x B, Y, X, F)
        feature_per_frame_pair = NConvToBatch2D(compensate_gradient = True, n_conv = n_frame_pairs, inference_conv_idx=frame_window-1, filters = feature_filters, name = f"TrackingFeatures")(combined_features) # (N_OUT x B, Y, X, F)

        for decoder_name, is_segmentation in decoder_is_segmentation.items():
            if is_segmentation is not None:
                d_layers = decoder_layers[decoder_name]
                for output_name in output_per_decoder[decoder_name]:
                    if output_name in decoder_out[decoder_name]:
                        d_out = decoder_out[decoder_name][output_name]
                        layer_output_name = decoder_output_names[decoder_name][output_name]
                        skip = skip_per_decoder[decoder_name]
                        up = feature_per_frame if is_segmentation else feature_per_frame_pair
                        for op in decoder_feature_op[decoder_name][0]:
                            up = op(up)
                        for l, res in zip(d_layers[::-1], residuals[:-1]):
                            up = l([up, res if skip else None])
                        up = d_out([up, residuals[-1] if skip else None]) # (N_OUT x B, Y, X, F)
                        up = BatchToChannel2D(n_splits = n_chan if is_segmentation else n_frame_pairs, compensate_gradient = False, name = layer_output_name)(up)
                        outputs.append(up)
        return DistnetModel([input], outputs, name=name, frame_window=frame_window, next = next, predict_contours = False, predict_center=predict_center, spatial_dims=spatial_dims, long_term=long_term, category_background=category_background, **kwargs)

def encoder_op(param_list, downsampling_mode, skip_stop_gradient:bool = False, last_input_filters:int=0, name: str="EncoderLayer", layer_idx:int=1):
    name=f"{name}{layer_idx}"
    maxpool = downsampling_mode=="maxpool"
    maxpool_and_stride = downsampling_mode == "maxpool_and_stride"
    sequence, down_sequence, total_contraction, residual_filters, out_filters = parse_param_list(param_list, name, ignore_stride=maxpool, last_input_filters = last_input_filters if maxpool_and_stride else 0)
    assert total_contraction>1, "invalid parameters: no contraction specified"
    if maxpool:
        down_sequence = []
    if maxpool or maxpool_and_stride:
        down_sequence = down_sequence+[MaxPool2D(pool_size=total_contraction, name=f"{name}/Maxpool{total_contraction}x{total_contraction}")]
        down_concat = tf.keras.layers.Concatenate(axis=-1, name = f"{name}/DownConcat", dtype="float32")
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
        if sequence is not None or layer_idx>0:
            res = x
            if skip_stop_gradient:
                res = stop_gradient(res, parent_name = name)
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
            factor:float = 1,
            name: str="DecoderLayer",
            layer_idx:int=1,
        ):
        name=f"{name}{layer_idx}"
        if n_conv==0 and activation_out is not None:
            activation = activation_out
        if n_conv>0 and activation_out is None:
            activation_out = activation
        if n_conv==0 and filters_out is not None:
            filters = filters_out
        if n_conv>0 and filters_out is None:
            filters_out = filters

        up_op = upsampling_op(filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, weight_scaled = weight_scaled_up, batch_norm=batch_norm_up, dropout_rate=dropout_rate_up)
        if skip_combine_mode.lower()=="conv" or skip_combine_mode.lower()=="wsconv":
            combine = Combine(name = name, filters=filters, kernel_size = combine_kernel_size, weight_scaled=skip_combine_mode.lower()=="wsconv")
        else:
            combine = None
        op = op.lower().replace("_", "")
        if op == "res1d" or op=="resconv1d":
            convs = [ResConv1D(kernel_size=conv_kernel_size, activation=activation_out if i==n_conv-1 else activation, weight_scaled=weight_scaled, batch_norm=batch_norm, dropout_rate=dropout_rate, weighted_sum=weighted_sum, name=f"{name}/ResConv1D_{i}_{conv_kernel_size}x{conv_kernel_size}") for i in range(n_conv)]
        elif op == "res2d" or op=="resconv2d":
            convs = [ResConv2D(kernel_size=conv_kernel_size, activation=activation_out if i==n_conv-1 else activation, weight_scaled=weight_scaled, batch_norm=batch_norm, dropout_rate=dropout_rate, weighted_sum=weighted_sum, name=f"{name}/ResConv2D_{i}_{conv_kernel_size}x{conv_kernel_size}") for i in range(n_conv)]
        else:
            if weight_scaled:
                convs = [WSConv2D(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, padding='same', activation=activation_out if i==n_conv-1 else activation, dropout_rate=dropout_rate, name=f"{name}/Conv_{i}_{conv_kernel_size}x{conv_kernel_size}") for i in range(n_conv)]
            elif batch_norm or dropout_rate>0:
                convs = [Conv2DBNDrop(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, activation=activation_out if i==n_conv-1 else activation, batch_norm=batch_norm, dropout_rate=dropout_rate, name=f"{name}/Conv_{i}_{conv_kernel_size}x{conv_kernel_size}") for i in range(n_conv)]
            else:
                convs = [Conv2D(filters=filters_out if i==n_conv-1 else filters, kernel_size=conv_kernel_size, padding='same', activation=activation_out if i==n_conv-1 else activation, name=f"{name}/Conv_{i}_{conv_kernel_size}x{conv_kernel_size}") for i in range(n_conv)]
        f = tf.cast(factor, tf.float32)
        def op(input):
            down, res = input
            up = up_op(down)
            if res is not None:
                if combine is not None:
                    up = combine([up, res])
                else:
                    up = up + res
            for c in convs:
                up = c(up)
            if factor!=1:
                up = up * f
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
            name: str="Upsampling2D",
        ):
        assert mode in ["tconv", "up_nn", "up_bilinear"], "invalid mode"
        if kernel_size<size_factor:
            kernel_size = size_factor
        if parent_name is not None and len(parent_name)>0:
            name = f"{parent_name}/{name}"
        if mode=="tconv":
            if weight_scaled:
                upsample = WSConv2DTranspose(filters=filters, kernel_size=kernel_size, strides=size_factor, activation=activation, dropout_rate=dropout_rate, padding='same', name=f"{name}/tConv{kernel_size}x{kernel_size}")
            elif batch_norm or dropout_rate>0:
                upsample = Conv2DTransposeBNDrop(filters=filters, kernel_size=kernel_size, strides=size_factor, activation=activation, batch_norm=batch_norm, dropout_rate=dropout_rate, name=f"{name}/tConv{kernel_size}x{kernel_size}")
            else:
                upsample = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides=size_factor, padding='same', activation=activation, use_bias=use_bias, name=f"{name}/tConv{kernel_size}x{kernel_size}")
            conv=None
        else:
            interpolation = "nearest" if mode=="up_nn" else 'bilinear'
            upsample = tf.keras.layers.UpSampling2D(size=size_factor, interpolation=interpolation, name = f"{name}/Upsample{size_factor}x{size_factor}_{interpolation}")
            if batch_norm:
                conv = Conv2DBNDrop(filters=filters, kernel_size=kernel_size, strides=1, batch_norm=batch_norm, dropout_rate=dropout_rate, name=f"{name}/Conv{kernel_size}x{kernel_size}", activation=activation )
            else:
                conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding='same', name=f"{name}/Conv{kernel_size}x{kernel_size}", use_bias=use_bias, activation=activation )
        def op(input):
            x = upsample(input)
            if conv is not None:
                x = conv(x)
            return x
        return op

def stop_gradient(input, parent_name:str, name:str="StopGradient"):
    if parent_name is not None and len(parent_name)>0:
        name = f"{parent_name}/{name}"
    return tf.stop_gradient( input, name=name )

def parse_param_list(param_list, name:str, last_input_filters:int=0, ignore_stride:bool = False):
    if param_list is None or len(param_list)==0:
        return [], None, 1, 0, 0
    total_contraction = 1
    if ignore_stride:
        param_list = [params.copy() for params in param_list]
        for params in param_list:
            total_contraction *= params.get("downscale", 1)
            params["downscale"] = 1
    # split into squence with no stride (for residual) and the rest of the sequence
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
            sequence.append(parse_params(**param_list[i], name = f"{name}/Op{i}"))
            i+=1
    else:
        sequence=None
        residual_filters = 0
    if i<len(param_list):
        if i==len(param_list)-1:
            params = param_list[i].copy()
            filters = params.pop("filters")
            out_filters = filters
            if last_input_filters>0: # case of stride + maxpool -> out filters -= input filters
                if residual_filters>0:
                    last_input_filters=residual_filters # input of downscaler is the residual
                filters -= last_input_filters
            down = [parse_params(**params, filters=filters, name=f"{name}/DownOp")]
            total_contraction *= param_list[i].get("downscale", 1)
        else:
            raise ValueError("Only one downscale operation allowed")
    else:
        down = None
        out_filters = residual_filters
    return sequence, down, total_contraction, residual_filters, out_filters

def parse_params(filters:int = 0, kernel_size:int = 3, op:str = "conv", dilation:int=1, activation="relu", downscale:int=1, dropout_rate:float=0, weight_scaled:bool=False, batch_norm:bool=False, weighted_sum:bool=False, name:str=""):
    op = op.lower().replace("_", "")
    if op =="res1d" or op=="resconv1d":
        return ResConv1D(kernel_size=kernel_size, dilation=dilation, activation=activation, dropout_rate=dropout_rate, weight_scaled = weight_scaled, batch_norm=batch_norm, weighted_sum=weighted_sum, name=f"{name}/ResConv1D{kernel_size}x{kernel_size}")
    elif op =="res2d" or op == "resconv2d":
        return ResConv2D(kernel_size=kernel_size, dilation=dilation, activation=activation, dropout_rate=dropout_rate, weight_scaled=weight_scaled, batch_norm=batch_norm, weighted_sum=weighted_sum, name=f"{name}/ResConv2D{kernel_size}x{kernel_size}")
    assert filters > 0 , "filters must be > 0"
    if op=="selfattention":
        self_attention_op = SpatialAttention2D(positional_encoding="2D", name=f"{name}/SelfAttention")
        self_attention_skip_op = Combine(filters=filters, name=f"{name}/SelfAttentionSkip")
        def op(x):
            sa = self_attention_op([x, x])
            return self_attention_skip_op([x, sa])
        return op
    if weight_scaled:
        return WSConv2D(filters=filters, kernel_size=kernel_size, strides = downscale, dilation_rate = dilation, activation=activation, dropout_rate=dropout_rate, padding='same', name=f"{name}/Conv{kernel_size}x{kernel_size}")
    elif batch_norm or dropout_rate>0:
        return Conv2DBNDrop(filters=filters, kernel_size=kernel_size, strides = downscale, dilation = dilation, activation=activation, dropout_rate=dropout_rate, batch_norm=batch_norm, name=f"{name}/Conv{kernel_size}x{kernel_size}")
    else:
        return Conv2D(filters=filters, kernel_size=kernel_size, strides = downscale, dilation_rate = dilation, padding='same', activation=activation, name=f"{name}/Conv{kernel_size}x{kernel_size}")
