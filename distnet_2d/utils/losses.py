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

def edm_contour_loss(original_loss_func, background_weight, edm_weight, contour_weight, dtype='float32'):
    '''
    This function allows to set a distinct weight for contour values (edm==1) and rest of foreground and background
    '''
    weights_values = np.array((background_weight, edm_weight, contour_weight)).astype(dtype)
    def loss_func(y_true, y_pred):
        weight_map = tf.where(y_true==0, weights_values[0], tf.where(y_true==1, weights_values[2], weights_values[1]))
        loss = original_loss_func(y_true, y_pred)
        loss = loss * weight_map
        return loss
    return loss_func
