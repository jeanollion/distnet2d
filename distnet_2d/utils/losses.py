import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def ssim_loss(max_val = 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    def loss_fun(y_true, y_pred):
        SSIM = tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        return 1 - (1 + SSIM ) * 0.5
    return loss_fun

def weighted_loss_by_category(original_loss_func, weight_list, axis=-1, sparse=True, dtype='float32'):
    weight_list = np.array(weight_list).astype("float32")
    # normalize weights:
    n_classes = weight_list.shape[0]
    weight_list = n_classes * weight_list / np.sum(weight_list)
    weight_list = tf.cast(weight_list, dtype=dtype)
    def loss_func(y_true, y_pred):
        if sparse:
            class_weights = tf.squeeze(y_true, axis=-1)
            if not class_weights.dtype.is_integer:
                class_weights = tf.cast(class_weights, tf.int32)
            class_weights = tf.one_hot(class_weights, n_classes, dtype=dtype)
        else:
            class_weights = tf.cast(y_true, dtype=dtype)
        class_weights = tf.reduce_sum(class_weights * weight_list, axis=-1, keepdims=False) # multiply with broadcast

        loss = original_loss_func(y_true, y_pred)
        loss = loss * class_weights
        return loss
    return loss_func

def balanced_category_loss(original_loss_func, n_classes, max_class_ratio=20, no_background=False, axis=-1, sparse=True, add_channel_axis=False, dtype='float32'):
    weight_limits = np.array([1./max_class_ratio, max_class_ratio]).astype(dtype)
    def loss_func(y_true, y_pred):
        if sparse:
            class_weights = tf.squeeze(y_true, axis=-1)
            if not class_weights.dtype.is_integer:
                class_weights = tf.cast(class_weights, tf.int32)
            if no_background:
                class_weights = tf.one_hot(class_weights, n_classes+1, dtype=dtype)
                class_weights = class_weights[..., 1:] # remove background class
            else:
                class_weights = tf.one_hot(class_weights, n_classes, dtype=dtype)
        else:
            class_weights = tf.cast(y_true, dtype=dtype)

        count = tf.cast(tf.size(y_true), dtype=tf.float32)
        class_count = tf.math.count_nonzero(class_weights, axis=tf.range(tf.rank(class_weights)-1), dtype=tf.float32)
        class_count = tf.where(class_count==0, tf.reduce_max(class_count), class_count)
        # zero -> max weight ?
        weight_list = tf.math.divide_no_nan(count, class_count)
        weight_list = tf.math.minimum(weight_limits[1], tf.math.maximum(weight_limits[0], weight_list))
        weight_list = tf.math.divide_no_nan(weight_list, tf.math.reduce_sum(weight_list) / tf.cast(n_classes, dtype=tf.float32)) # normalize so that sum of weights == n_classes
        weight_list = tf.cast(weight_list, dtype=dtype)
        #print(f"class weights: {weight_list.numpy()}")
        class_weights = tf.reduce_sum(class_weights * weight_list, axis=-1, keepdims=False) # multiply with broadcast

        loss = original_loss_func(y_true, y_pred)
        loss = loss * class_weights
        return loss
    return loss_func

def weighted_binary_crossentropy(weights, add_channel_axis=True, **bce_kwargs):
    weights_cast = np.array(weights).astype("float32")
    assert weights_cast.shape[0] == 2, f"invalid weight number: expected: 2 given: {weights_cast.shape[0]}"
    bce = tf.keras.losses.BinaryCrossentropy(**bce_kwargs)
    def loss_func(true, pred):
        weights = tf.where(true, weights_cast[1], weights_cast[0])
        if add_channel_axis:
            true = tf.expand_dims( true, -1)
            pred = tf.expand_dims( pred, -1)
        else:
            weights = tf.squeeze(weights, axis=-1)
        return bce(true, pred, sample_weight=weights)
    return loss_func

def balanced_displacement_loss(dMin, dMax, wMin, wMax, l2=True, add_channel_axis=True):
    assert dMin<dMax, "expected dMin < dMax"
    params = np.array([dMin, dMax, wMin, wMax]).astype("float32")
    a = ( params[3] - params[2] ) / ( params[1] - params[0] )
    loss = tf.keras.losses.MeanSquaredError() if l2 else tf.keras.losses.MeanAbsoluteError()
    def loss_func(true, pred):
        weights = tf.where(true<=params[0], params[2], params[3])
        weights = tf.where(tf.logical_and(true>params[0], true<params[1]), a * (true - params[0]) , weights)
        if add_channel_axis:
            true = tf.expand_dims( true, -1)
            pred = tf.expand_dims( pred, -1)
        else:
            weights = tf.squeeze(weights, axis=-1)
        return loss(true, pred, sample_weight=weights)
    return loss_func

def edm_contour_loss(background_weight, edm_weight, contour_weight, l1=False, dtype='float32'):
    '''
    This function allows to set a distinct weight for contour values (edm==1) and rest of foreground and background
    '''
    weights_values = np.array((background_weight, edm_weight, contour_weight)).astype(dtype)
    if contour_weight>0:
        def loss_func(y_true, y_pred):
            weight_map = tf.where(y_true==0, weights_values[0], tf.where(y_true==1, weights_values[2], weights_values[1]))
            loss = tf.math.square(y_true - y_pred) if not l1 else tf.math.abs(y_true - y_pred)
            loss = loss * weight_map
            return tf.reduce_mean(loss, -1)
    else:
        def loss_func(y_true, y_pred):
            weight_map = tf.where(y_true==0, weights_values[0], weights_values[1])
            loss = tf.math.square(y_true - y_pred) if not l1 else tf.math.abs(y_true - y_pred)
            loss = loss * weight_map
            return tf.reduce_mean(loss, -1)
    return loss_func

def balanced_background_binary_crossentropy(add_channel_axis=True, max_class_ratio=100, **loss_kwargs):
    return balanced_background_loss(tf.keras.losses.BinaryCrossentropy(**loss_kwargs), add_channel_axis, True, max_class_ratio)

def balanced_background_l_norm(l2=True, add_channel_axis=True, max_class_ratio=100, **loss_kwargs):
    return balanced_background_loss(tf.keras.losses.MeanSquaredError(**loss_kwargs) if l2 else tf.keras.losses.MeanAbsoluteError(**loss_kwargs), add_channel_axis, False, max_class_ratio)

def balanced_background_loss(loss, add_channel_axis=True, y_true_bool = False, max_class_ratio=100):
    def loss_func(y_true, y_pred, sample_weight=None):
        weight_map = _compute_background_weigh_map(y_true, y_true_bool, max_class_ratio)
        if add_channel_axis:
            y_true = tf.expand_dims( y_true, -1)
            y_pred = tf.expand_dims( y_pred, -1)
        else:
            weight_map = tf.squeeze(weight_map, axis=-1)
        if sample_weight is not None:
            weight_map = sample_weight * weight_map
        return loss(y_true, y_pred, sample_weight=weight_map)
    return loss_func

def _compute_background_weigh_map(y_true, bool=False, max_class_ratio=100):
    fore_count = tf.math.count_nonzero(y_true, dtype=tf.float32)
    count = tf.cast(tf.size(y_true), dtype=tf.float32)
    weight_limits = np.array([1./max_class_ratio, max_class_ratio]).astype('float32')
    fore_w = count / fore_count
    bck_w = count / (count - fore_count)
    fore_w = tf.math.minimum(weight_limits[1], tf.math.maximum(weight_limits[0], fore_w))
    bck_w = tf.math.minimum(weight_limits[1], tf.math.maximum(weight_limits[0], bck_w))

    norm = 0.5 * (fore_w + bck_w) # scale the loss so that sum of weights == class number
    fore_w = fore_w / norm
    bck_w = bck_w / norm
    return tf.where(y_true, fore_w, bck_w) if bool else tf.where(y_true==tf.cast(0, y_true.dtype), bck_w, fore_w)
