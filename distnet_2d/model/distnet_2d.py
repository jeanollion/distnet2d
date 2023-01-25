import tensorflow as tf
from .layers import ConvNormAct, Bneck, UpSamplingLayer2D, StopGradient, Combine, WeigthedGradient
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
import numpy as np
from .self_attention import SelfAttention
from .attention import Attention
from .directional_2d_self_attention import Directional2DSelfAttention
from ..utils.helpers import ensure_multiplicity, flatten_list
from .utils import get_layer_dtype
from ..utils.losses import weighted_binary_crossentropy, weighted_loss_by_category, balanced_category_loss, edm_contour_loss, balanced_background_binary_crossentropy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, MeanSquaredError
from .coordinate_op_2d import get_soft_argmax_2d_fun, get_weighted_mean_2d_fun, GaussianSpread

ENCODER_SETTINGS = [
    [ # l1 = 128 -> 64
        {"filters":16},
        {"filters":32, "downscale":2}
    ],
    [  # l2 64 -> 32
        {"filters":32, "kernel_size":5},
        {"filters":32},
        {"filters":64, "downscale":2}
    ],
    [ # l3: 32->16
        {"filters":64, "kernel_size":5},
        {"filters":64, "kernel_size":5},
        {"filters":64},
        {"filters":128, "downscale":2},
    ],
]
FEATURE_SETTINGS = [
    {"filters":128, "kernel_size":5},
    {"filters":1024},
]

DECODER_SETTINGS = [
    {"filters":64}, # 96 ?
    {"filters":256},
    {"filters":512}
]

DECODER_SETTINGS_DS = [
    #f, s
    {"filters":16},
    {"filters":96},
    {"filters":128},
    {"filters":256}
]

class DistnetModel(Model):
    def __init__(self, *args, spatial_dims,
        edm_loss_weight=1, contour_loss_weight=1, displacement_loss_weight=1, category_loss_weight=1, displacement_var_weight=1./10, displacement_var_max=50, edm_loss=MeanSquaredError(),
        contour_loss = MeanSquaredError(),
        center_loss = MeanSquaredError(),
        displacement_loss = MeanSquaredError(),
        displacement_mean = False,
        category_weights = None, # array of weights: [background, normal, division, no previous cell] or None = auto
        category_class_frequency_range=[1/10, 10],
        next = True,
        frame_window = 1,
        predict_contours = True,
        predict_center = False,
        center_mode = "EDM_MAX",
        center_softargmax_beta = 1e2,
        center_sigma = 4,
        contour_sigma = 1,
        **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_contours = predict_contours
        self.predict_center = predict_center
        self.spatial_dims = spatial_dims
        if self.predict_center:
            self.center_loss=center_loss
            center_softargmax_beta = center_softargmax_beta
            self.center_spead = GaussianSpread(center_sigma, spatial_dims[0], spatial_dims[1], objectwise=True)
            self.get_center=get_weighted_mean_2d_fun()  # TODO: compare softmax vs mean
            self.center_displacement_loss = MeanSquaredError()
            self.center_mode=center_mode
            if center_mode == "EDM_MAX":
                self.edm_to_center = get_soft_argmax_2d_fun(beta=center_softargmax_beta)
            elif center_mode == "EDM_MEAN":
                self.edm_to_center = get_weighted_mean_2d_fun()
            else:
                self.edm_to_center = None
            if self.edm_to_center is not None:
                self.edm_center_loss = MeanSquaredError()
        self.contour_sigma = contour_sigma
        if self.contour_sigma>0:
            self.contour_edm_loss = MeanSquaredError()
        self.next = next
        self.frame_window = frame_window
        self.update_loss_weights(edm_loss_weight, contour_loss_weight, displacement_loss_weight, category_loss_weight)
        self.displacement_var_weight=displacement_var_weight
        self.displacement_var_max=displacement_var_max
        self.edm_loss = edm_loss
        min_class_frequency=category_class_frequency_range[0]
        max_class_frequency=category_class_frequency_range[1]
        self.contour_loss = contour_loss
        if category_weights is not None:
            assert len(category_weights)==4, "4 category weights should be provided: background, normal cell, dividing cell, cell with no previous cell"
            self.category_loss=weighted_loss_by_category(SparseCategoricalCrossentropy(), category_weights)
        else:
            self.category_loss = balanced_category_loss(SparseCategoricalCrossentropy(), 4, min_class_frequency=min_class_frequency, max_class_frequency=max_class_frequency) # TODO optimize this: use CategoricalCrossentropy to avoid transforming to one_hot twice
        self.displacement_loss = displacement_loss
        self.displacement_mean=displacement_mean


    def update_loss_weights(self, edm_weight=1, contour_weight=1, center_weight=1, displacement_weight=1, category_weight=1, normalize=True): # TODO: normalize per decoder head ?
        sum = edm_weight + (contour_weight if self.predict_contours else 0) + (center_weight if self.predict_center else 0) + displacement_weight + category_weight if normalize else 1
        self.edm_weight = edm_weight / sum
        self.contour_weight=contour_weight / sum
        self.center_weight = center_weight / sum
        self.displacement_weight=displacement_weight / sum
        self.category_weight=category_weight / sum

    def train_step(self, data):
        fw = self.frame_window
        mixed_precision = tf.keras.mixed_precision.global_policy().name == "mixed_float16"
        x, y = data
        displacement_weight = self.displacement_weight / 2. # y & x
        displacement_var_weight = self.displacement_var_weight
        category_weight = self.category_weight / (self.frame_window * (2. if self.next else 1))
        contour_weight = self.contour_weight
        edm_weight = self.edm_weight
        center_weight = self.center_weight
        inc = 1 if self.predict_contours else 0
        inc += 1 if self.predict_center else 0
        if len(y) == 6 + inc: # y = edm, contour, center, dy, dX, cat, no_next, label_rank
            label_rank, label_size = self._get_label_rank_and_size(y[-1]) # (B, Y, X, C, T, N) & (B, 1, 1, C, T, N)
            if self.displacement_mean:
                displacement_weight = displacement_weight / 2.
        else :
            label_rank = None
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # compute loss
            losses = dict()
            inc=0
            edm_loss = self.edm_loss(y[inc], y_pred[inc])
            loss = edm_loss * edm_weight
            losses["edm"] = tf.reduce_mean(edm_loss)

            if self.predict_contours:
                inc+=1
                contour_loss = self.contour_loss(y[inc], y_pred[inc])
                loss = loss + contour_loss * contour_weight
                losses["contour"] = tf.reduce_mean(contour_loss)
                # contour coherence
                if self.contour_sigma>0:
                    mul = -1./(self.contour_sigma * self.contour_sigma)
                    zero = tf.cast(0, tf.float32)
                    edm_c = tf.where(tf.math.greater(y[0], zero), tf.math.exp(tf.math.square(y_pred[0]-1) * mul), zero)
                    contour_edm_loss = self.contour_edm_loss(y_pred[1], edm_c)
                    loss = loss + contour_edm_loss * contour_weight
                    losses["contour_edm"] = tf.reduce_mean(contour_edm_loss)

            if self.predict_center:
                inc+=1
                center_loss = self.center_loss(y[inc], y_pred[inc])
                loss = loss + center_loss * center_weight
                losses["center"] = tf.reduce_mean(center_loss)
            # displacement loss
            dy_t = tf.expand_dims(y[1+inc], -1)
            dx_t = tf.expand_dims(y[2+inc], -1)
            if label_rank is not None: # label rank is returned : object-wise loss
                label_rank_sel = tf.concat([tf.tile(label_rank[..., fw:fw+1, :], [1,1,1,fw,1]), label_rank[..., fw+1:, :]], axis=-2)
                label_size_sel = tf.concat([tf.tile(label_size[..., fw:fw+1, :], [1,1,1,fw,1]), label_size[..., fw+1:, :]], axis=-2)
                dym_pred_center, dym_pred = self._get_mean_by_object(y_pred[1+inc], label_rank_sel, label_size_sel)
                dxm_pred_center, dxm_pred = self._get_mean_by_object(y_pred[2+inc], label_rank_sel, label_size_sel)
                if self.predict_center: # center+displacement coherence loss
                    # predicted  center coord per object
                    nan = tf.cast(float('NaN'), tf.float32)
                    d_pred_center = tf.stack([dym_pred_center, dxm_pred_center], -1) # (B, 1, 1, T-1, N, 2)
                    center_pred = y_pred[inc] # (B, Y, X, T)
                    center_pred_ob = self._get_center(center_pred, self.get_center, label_rank) # (B, 1, 1, T, N, 2)
                    no_prev = tf.equal(self._get_mean_by_object(y[3+inc], label_rank_sel, label_size_sel, project=False), tf.cast(3, tf.float32))[...,tf.newaxis] # (B, 1, 1, T-1, N, 1)
                    no_next = y[-2][...,:label_rank.shape.as_list()[-1]] # (B, T-1, N) # trim no_next from (B, T-1, n_label_max)
                    no_next = no_next[:, tf.newaxis, tf.newaxis, :, :, tf.newaxis] # (B, 1, 1, T-1, N, 1)
                    # current -> previous:  move cur so that it corresponds to prev
                    center_pred_ob_cur = center_pred_ob[...,fw:fw+1,:,:]
                    center_pred_ob_cur_to_prev = center_pred_ob_cur - d_pred_center[...,:fw,:,:] # (B, 1, 1, FW, N, 2)
                    # remove centers with no prev
                    center_pred_ob_cur_to_prev = tf.where(no_prev[...,:fw, :, :], nan, center_pred_ob_cur_to_prev)
                    center_pred_ob_cur_to_prev = self.center_spead(center_pred_ob_cur_to_prev) # (B, Y, X, FW)
                    # target is previous centers excluding those with no_next
                    center_pred_ob_prev = center_pred_ob[...,:fw,:,:]
                    center_pred_ob_prev = tf.where(no_next[...,:fw, :, :], nan, center_pred_ob_prev)
                    center_pred_ob_prev = self.center_spead(center_pred_ob_prev) # (B, Y, X, FW)
                    center_displacement_loss = self.center_displacement_loss(center_pred_ob_prev, center_pred_ob_cur_to_prev) # (B, Y, X, FW)

                    if self.next:
                        # next -> current : move next so that it corresponds to current
                        center_pred_ob_next = center_pred_ob[...,fw+1:,:,:] # (B, 1, 1, FW, N, 2)
                        center_pred_ob_next_to_cur = center_pred_ob_next - d_pred_center[...,fw:,:,:] # (B, 1, 1, FW, N, 2)
                        # remove centers with no prev
                        center_pred_ob_next_to_cur = tf.where(no_prev[...,fw:, :, :], nan, center_pred_ob_next_to_cur)
                        center_pred_ob_next_to_cur = self.center_spead(center_pred_ob_next_to_cur) # (B, Y, X, FW)
                        # target is current centers excluding those with no_next
                        center_pred_ob_cur = tf.where(no_next[...,fw:, :, :], nan, center_pred_ob_cur)
                        center_pred_ob_cur = self.center_spead(center_pred_ob_cur) # (B, Y, X, FW)
                        center_displacement_loss += self.center_displacement_loss(center_pred_ob_cur, center_pred_ob_next_to_cur)

                    loss = loss + center_displacement_loss * (center_weight / (2. if self.next else 1.))
                    losses["center_displacement"] = tf.reduce_mean(center_displacement_loss)

                    if self.edm_to_center is not None: # center edm loss
                        edm_center_ob = self._get_center(y_pred[0], self.edm_to_center, label_rank) # (B, 1, 1, T, N, 2)
                        edm_center_ob = self.center_spead(edm_center_ob) # (B, Y, X, T)
                        edm_center_loss = self.edm_center_loss(center_pred, edm_center_ob)
                        loss = loss + edm_center_loss * center_weight
                        losses["edm_center"] = tf.reduce_mean(edm_center_loss)

                if self.displacement_var_weight>0: # enforce homogeneity : increase weight
                    _, dy2m_pred = self._get_mean_by_object(tf.math.square(y_pred[1+inc]), label_rank_sel, label_size_sel)
                    vary = dy2m_pred - tf.math.square(dym_pred)
                    _, dx2m_pred = self._get_mean_by_object(tf.math.square(y_pred[2+inc]), label_rank_sel, label_size_sel)
                    varx = dx2m_pred - tf.math.square(dxm_pred)
                    var = vary + varx
                    var = stop_gradient(var, parent_name = "DisplacementVAR") # we create a weight map -> no gradient
                    if self.displacement_var_max>0:
                        var = tf.math.minimum(var, self.displacement_var_max)
                    displacement_wm = 1 + var * displacement_var_weight
                else:
                    displacement_wm = None
                if self.displacement_mean:
                    dm_loss = self.displacement_loss(dy_t, tf.expand_dims(dym_pred, -1), sample_weight=displacement_wm) + self.displacement_loss(dx_t, tf.expand_dims(dxm_pred, -1), sample_weight=displacement_wm)
                    loss = loss + dm_loss * displacement_weight
                    losses["displacement_mean"] = tf.reduce_mean(dm_loss)
            else:
                displacement_wm = None
            # pixel-wise displacement loss
            d_loss = self.displacement_loss(dy_t, tf.expand_dims(y_pred[1+inc], -1), sample_weight=displacement_wm) + self.displacement_loss(dx_t, tf.expand_dims(y_pred[2+inc], -1), sample_weight=displacement_wm)
            loss = loss + d_loss * displacement_weight
            losses["displacement"] = tf.reduce_mean(d_loss)

            # category loss
            cat_loss = 0
            for i in range(self.frame_window * (2 if self.next else 1)):
                cat_loss = cat_loss + self.category_loss(y[3+inc][...,i:i+1], y_pred[3+inc][...,4*i:4*i+4])
            loss = loss + cat_loss * category_weight
            losses["category"] = tf.reduce_mean(cat_loss)
            losses["loss"] = tf.reduce_mean(loss)
            if mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)

        # Compute gradients
        trainable_vars = self.trainable_variables
        grad = tape.gradient(loss, trainable_vars)
        if mixed_precision:
            grad = self.optimizer.get_unscaled_gradients(grad)
        # Update weights
        self.optimizer.apply_gradients(zip(grad, trainable_vars))
        return losses

    def _get_center(self, center, center_fun, label_rank):
        center_ob = label_rank * tf.expand_dims(center, -1) # (B, Y, X, T, N)
        shape = tf.shape(center_ob)
        center_ob = tf.reshape(center_ob, tf.concat([shape[:3], [-1]], 0) ) #(B, Y, X, TxN)
        center_ob = center_fun(center_ob) # (B, 1, 1, TxN, 2)
        return tf.reshape(center_ob, tf.concat([shape[:1], [1, 1], shape[-2:], [2]],0)) # (B, 1, 1, T, N, 2)

    def _get_mean_by_object(self, data, label_rank, label_size, project=True):
        wsum = tf.reduce_sum(label_rank * tf.expand_dims(data, -1), axis=[1, 2], keepdims = True)
        mean = tf.math.divide_no_nan(wsum, label_size) # batch, 1, 1, C, n_label_max
        if project:
            mean_p = tf.reduce_sum(mean * label_rank, axis=-1) # batch, y, x, 1 or 2
            return mean, mean_p
        else:
            return mean

    def _get_label_rank_and_size(self, labels):
        N = tf.math.maximum(1, tf.math.reduce_max(labels)) # in case of empty image: 1
        label_rank = tf.one_hot(labels-1, N, dtype=tf.float32) # B, Y, X, T, N
        label_size = tf.reduce_sum(label_rank, axis=[1, 2], keepdims=True) # B, 1, 1, T, N
        return label_rank, label_size

# one encoder per input + one decoder + one last level of decoder per output + custom frame window size
def get_distnet_2d_sep_out_fw(input_shape, # Y, X
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            combine_kernel_size:int = 3,
            skip_stop_gradient:bool = False,
            predict_contours:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = DECODER_SETTINGS,
            residual_combine_size:int = 3,
            frame_window:int = 1,
            predict_center = False,
            next:bool=True,
            name: str="DiSTNet2D",
            l2_reg: float=1e-5,
    ):
        total_contraction = np.prod([np.prod([params.get("downscale", 1) for params in param_list]) for param_list in encoder_settings])
        assert len(encoder_settings)==len(decoder_settings), "decoder should have same length as encoder"
        spatial_dims = ensure_multiplicity(2, input_shape)
        n_chan = frame_window * (2 if next else 1) + 1
        # define enconder operations
        encoder_layers = []
        contraction_per_layer = []
        combine_residual_layer = []
        for l_idx, param_list in enumerate(encoder_settings):
            op, contraction, residual_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters, kernel_size=residual_combine_size, name=f"CombineResiduals{l_idx}"))
        # define feature operations
        feature_convs, _, _, attention_filters = parse_param_list(feature_settings, "FeatureSequence")
        attention_op = Attention(positional_encoding="2D", name="Attention")
        self_attention_op = Attention(positional_encoding="2D", name="SelfAttention")
        self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        combine_features_op = Combine(filters=attention_filters//2, name="CombineFeatures")
        attention_combine = Combine(filters=attention_filters//2, name="AttentionCombine")
        attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers=[]
        decoder_out = []
        output_inc = 0
        seg_out = ["Output0_EDM"]
        activation_out = ["linear"]
        if predict_contours:
            output_inc += 1
            seg_out += ["Output1_Contours"]
            activation_out += ["linear"]
        if predict_center:
            output_inc += 1
            seg_out += [f"Output{output_inc}_Center"]
            activation_out += ["linear"]

        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out.append( decoder_sep_op(**param_list, output_names =seg_out, name="DecoderSegmentation", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out=activation_out ))
                decoder_out.append( decoder_sep_op(**param_list, output_names = [f"Output{1+output_inc}_dy", f"Output{2+output_inc}_dx"], name="DecoderDisplacement", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu") )
                cat_names = [f"Output_Category_{i}" for i in range(0, frame_window)]
                if next:
                    cat_names += [f"Output_CategoryNext_{i}" for i in range(0, frame_window)]
                decoder_out.append( decoder_sep2_op(**param_list, output_names = cat_names, name="DecoderCategory", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="softmax", filters_out=4, output_name = f"Output{3+output_inc}_Category") ) # categories are concatenated
            else:
                decoder_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx) )

        # Create GRAPH
        input = tf.keras.layers.Input(shape=spatial_dims+(n_chan,), name="Input")
        inputs = tf.split(input, num_or_size_splits = n_chan, axis=-1)
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
        attention_result = []
        for i in range(0, frame_window):
            attention_result.append(attention_op([all_features[i], all_features[frame_window]]))
        if next:
            for i in range(0, frame_window):
                attention_result.append(attention_op([all_features[frame_window], all_features[frame_window+1+i]]))
        attention = attention_combine(attention_result)
        feature = attention_skip_op([attention, combined_features])

        residuals = []
        for l_idx in range(len(encoder_layers)): # arrange residuals and combine them except for first layer
            res = [residuals_c[l_idx] for residuals_c in all_residuals]
            #grad_weight_op = WeigthedGradient(1./3, name=f"WeigthedGradient_{l_idx}")
            if l_idx>=1:
                combine_residual_op = combine_residual_layer[l_idx]
                res = combine_residual_op(res)
            #res = grad_weight_op(res) #
            residuals.append(res)

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        last_residuals = residuals[-1]
        residuals_displacement = [last_residuals[frame_window]]*frame_window # previous is from central->prev
        if next:
            residuals_displacement+=last_residuals[frame_window+1:] # next are from next->central
        seg = decoder_out[0]([ upsampled[-1], last_residuals ])
        dy, dx = decoder_out[1]([ upsampled[-1], residuals_displacement ])
        cat = decoder_out[2]([ upsampled[-1], residuals_displacement ])
        outputs = flatten_list([seg, dy, dx, cat])
        return DistnetModel([input], outputs, name=name, next = next, predict_contours = predict_contours, predict_center=predict_center, frame_window=frame_window, spatial_dims=spatial_dims)

def encoder_op(param_list, downsampling_mode, skip_stop_gradient:bool = False, name: str="EncoderLayer", layer_idx:int=1):
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
        if skip_stop_gradient:
            res = stop_gradient(res, parent_name = name)
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
        conv = Conv2D(filters=filters, kernel_size=conv_kernel_size, padding='same', activation=activation, name=f"{name}/Conv{conv_kernel_size}x{conv_kernel_size}")
        def op(input):
            down, res = input
            up = up_op(down)
            if combine is not None:
                x = combine([up, res])
            else:
                x = up + res
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
        activation_out = ensure_multiplicity(len(output_names), activation_out)
        filters_out = ensure_multiplicity(len(output_names), filters_out)
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = lambda x : upsampling_block(x, filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True) # l2_reg=l2_reg
        combine_gen = lambda i: Combine(name = f"{name}/Combine{i}", filters=filters, kernel_size = combine_kernel_size) #, l2_reg=l2_reg
        conv_out = [Conv2D(filters=f, kernel_size=conv_kernel_size, padding='same', activation=a, dtype="float32", name=f"{name}/{output_name}") for output_name, a, f in zip(output_names, activation_out, filters_out)]
        concat_out = [tf.keras.layers.Concatenate(axis=-1, name = output_name, dtype="float32") for output_name, a in zip(output_names, activation_out)]
        id_out = [tf.keras.layers.Lambda(lambda x: x, name = output_name, dtype="float32") for output_name, a in zip(output_names, activation_out)]
        def op(input):
            down, res_list = input
            up = up_op(down)
            if not isinstance(res_list, (list, tuple)):
                res_list = [res_list]
            x_list = [combine_gen(i)([up, res]) for i, res in enumerate(res_list)]
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
            output_name:str = None, # if provided : concat all outputs in one
            name: str="DecoderSepLayer",
            layer_idx:int=-1,
        ):
        if layer_idx>=0:
            name=f"{name}{layer_idx}"
        up_op = lambda x : upsampling_block(x, filters=filters, parent_name=name, size_factor=size_factor, kernel_size=up_kernel_size, mode=mode, activation=activation, use_bias=True) # l2_reg=l2_reg
        combine = [Combine(name = f"{name}/Combine{i}", filters=filters, kernel_size = combine_kernel_size) for i, _ in enumerate(output_names) ] #, l2_reg=l2_reg
        conv_out = [Conv2D(filters=filters_out, kernel_size=conv_kernel_size, padding='same', activation=activation_out, name=output_name, dtype='float32') for output_name in output_names]
        if output_names is not None:
            output_concat = tf.keras.layers.Concatenate(axis=-1, name = output_name, dtype="float32")
        def op(input):
            down, res_list = input
            assert len(res_list)==len(output_names), "decoder_sep2 : expected as many outputs as residuals"
            up = up_op(down)
            output_list = [ conv_out[i](combine[i]([up, res])) for i, res in enumerate(res_list) ]
            if output_name is not None:
                return output_concat(output_list)
            else:
                return output_list
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

#####################################################################
#################### LEGACY VERSION #################################
#####################################################################
# one encoder per input + one decoder + one last level of decoder per output
def get_distnet_2d_sep_out(input_shape,
            upsampling_mode:str="tconv", # tconv, up_nn, up_bilinear
            downsampling_mode:str = "stride", #maxpool, stride
            combine_kernel_size:int = 3,
            skip_stop_gradient:bool = False,
            predict_contours:bool = False,
            encoder_settings:list = ENCODER_SETTINGS,
            feature_settings: list = FEATURE_SETTINGS,
            decoder_settings: list = DECODER_SETTINGS,
            residual_combine_size:int = 1,
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
            op, contraction, residual_filters = encoder_op(param_list, downsampling_mode=downsampling_mode, skip_stop_gradient=skip_stop_gradient, layer_idx = l_idx)
            encoder_layers.append(op)
            contraction_per_layer.append(contraction)
            combine_residual_layer.append(Combine(filters=residual_filters * (3 if next else 2), kernel_size=residual_combine_size, name=f"CombineResiduals{l_idx}"))
        # define feature operations
        feature_convs, _, _, attention_filters = parse_param_list(feature_settings, "FeatureSequence")
        attention_op = Attention(positional_encoding="2D", name="Attention")
        self_attention_op = Attention(positional_encoding="2D", name="SelfAttention")
        self_attention_skip_op = Combine(filters=attention_filters, name="SelfAttentionSkip")
        combine_features_op = Combine(filters=attention_filters//2, kernel_size=residual_combine_size, name="CombineFeatures")
        attention_skip_op = Combine(filters=attention_filters//2, name="AttentionSkip")

        # define decoder operations
        decoder_layers=[]
        decoder_out = []
        output_inc = 1 if predict_contours else 0
        for l_idx, param_list in enumerate(decoder_settings):
            if l_idx==0:
                decoder_out.append( decoder_sep_op(**param_list, output_names = ["Output0_EDM", "Output1_Contours"] if predict_contours else ["Output0_EDM"], name="DecoderEDM", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out=["linear", "sigmoid"] if predict_contours else "linear") )
                decoder_out.append( decoder_sep_op(**param_list, output_names = [f"Output{1+output_inc}_dy", f"Output{2+output_inc}_dx"], name="DecoderDisplacement", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu") )
                cat_names = [f"Output{3+output_inc}_Category", f"Output{4+output_inc}_CategoryNext"] if next else [f"Output{3+output_inc}_Category"]
                decoder_out.append( decoder_sep2_op(**param_list, output_names = cat_names, name="DecoderCategory", size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, combine_kernel_size=combine_kernel_size, mode=upsampling_mode, activation="relu", activation_out="softmax", filters_out=4) )
            else:
                decoder_layers.append( decoder_op(**param_list, size_factor=contraction_per_layer[l_idx], conv_kernel_size=3, mode=upsampling_mode, skip_combine_mode="conv", combine_kernel_size=combine_kernel_size, activation="relu", layer_idx=l_idx) )

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
            #grad_weight_op = WeigthedGradient(1./3, name=f"WeigthedGradient_{l_idx}")
            if l_idx>=1:
                combine_residual_op = combine_residual_layer[l_idx]
                res = combine_residual_op(res)
            #res = grad_weight_op(res) #
            residuals.append(res)

        upsampled = [feature]
        residuals = residuals[::-1]
        for i, l in enumerate(decoder_layers[::-1]):
            up = l([upsampled[-1], residuals[i]])
            upsampled.append(up)

        last_residuals = residuals[-1]
        seg = decoder_out[0]([ upsampled[-1], last_residuals ])
        dy, dx = decoder_out[1]([ upsampled[-1], last_residuals[1:] ])
        cat = decoder_out[2]([ upsampled[-1], last_residuals[1:] ])
        outputs = flatten_list([seg, dy, dx, cat])
        return DistnetModel([input], outputs, name=name, next = next, predict_contours = predict_contours)
