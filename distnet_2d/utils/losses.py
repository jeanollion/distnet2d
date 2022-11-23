import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

def ssim_loss(max_val = 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    def loss_fun(y_true, y_pred):
        SSIM = tf.image.ssim(y_true, y_pred, max_val=max_val, filter_size=filter_size, filter_sigma=filter_sigma, k1=k1, k2=k2)
        return 1 - (1 + SSIM ) * 0.5
    return loss_fun

def weighted_loss_by_category(original_loss_func, weights_list, axis=-1, sparse=True, dtype='float32'):
    weights_list_cast = np.array(weights_list).astype(dtype)
    class_indices = np.array([i for i in range(len(weights_list_cast))]).astype(dtype)
    def loss_func(true, pred):
        if sparse:
            class_selectors = K.squeeze(true, axis=axis)
        else:
            class_selectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        class_selectors = [K.equal(i, class_selectors) for i in class_indices]

        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        class_selectors = [tf.cast(x, dtype) for x in class_selectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(class_selectors, weights_list_cast)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weight_multiplier = weights[0]
        for i in range(1, len(weights)):
            weight_multiplier = weight_multiplier + weights[i]

        #make sure your original_loss_func only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = original_loss_func(true, pred)
        loss = loss * weight_multiplier
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

def balanced_background_binary_crossentropy(add_channel_axis=True, **loss_kwargs):
    return balanced_background_loss(tf.keras.losses.BinaryCrossentropy(**loss_kwargs), add_channel_axis)

def balanced_background_l_norm(l2=True, add_channel_axis=True, **loss_kwargs):
    return balanced_background_loss(tf.keras.losses.MeanSquaredError(**loss_kwargs) if l2 else tf.keras.losses.MeanAbsoluteError(**loss_kwargs), add_channel_axis)

def balanced_background_loss(loss, add_channel_axis=True):
    def loss_func(y_true, y_pred):
        weight_map = _compute_background_weigh_map(y_true)
        if add_channel_axis:
            y_true = tf.expand_dims( y_true, -1)
            y_pred = tf.expand_dims( y_pred, -1)
        else:
            weight_map = tf.squeeze(weights, axis=-1)
        return loss(y_true, y_pred, sample_weight=weight_map)
    return loss_func

def _compute_background_weigh_map(y_true):
    fore_count = tf.math.count_nonzero(y_true, dtype=tf.dtypes.float32)
    count = tf.size(y_true, out_type=tf.dtypes.float32)
    fore_w = count / fore_count
    bck_w = count / (count - fore_count)
    return tf.where(y_true==0, bck_w, fore_w)
