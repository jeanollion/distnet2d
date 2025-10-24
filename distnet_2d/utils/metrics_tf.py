import tensorflow as tf
from .objectwise_computation_tf import get_max_by_object_fun, coord_distance_fun, get_argmax_2d_by_object_fun, \
    get_mean_by_object_fun, get_label_size, IoU, objectwise_compute, objectwise_compute_channel, reduce_pop_size, \
    FP


def get_metrics_fun(scale: float, max_objects_number: int = 0, category:bool = False, tracking:bool=True):
    """
    return metric function for disnet2D
    assumes iterator in return_central_only= True mode (thus framewindow = 1 and next = true)
    Parameters
    ----------
    scale
    max_objects_number
    reduce

    Returns function that inputs iterator output and distnet prediction and returns a tuple of 5 metric tensor, each tensor having as many elements as samples
    -------

    """

    coord_distance_function = coord_distance_fun(max=True, sqrt=True, pop_fraction=0.25)
    spa_max_fun = get_argmax_2d_by_object_fun()
    mean_fun = get_mean_by_object_fun()
    max_fun = get_max_by_object_fun(nan=1., channel_axis=False)
    mean_fun_true_lm = get_mean_by_object_fun(nan=1., channel_axis=False)
    mean_fun_lm = get_mean_by_object_fun(nan=0.)

    def fun(args):
        if category:
            if tracking:
                edm, gdcm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
            else:
                edm, gdcm, cat, true_edm, true_cat, labels, true_center_ob = args
        else:
            if tracking:
                edm, gdcm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
            else:
                edm, gdcm, true_edm, labels, true_center_ob = args
        labels = tf.transpose(labels, perm=[2, 0, 1])  # (1, Y, X)
        edm = tf.transpose(edm, perm=[2, 0, 1]) # (1, Y, X)
        gdcm = tf.transpose(gdcm, perm=[2, 0, 1])  # (1, Y, X)
        center_values = tf.math.exp(-tf.math.square(tf.math.divide(gdcm, tf.cast(scale/2., tf.float32))))
        ids, sizes, N = get_label_size(labels, max_objects_number)  # (1, N), (1, N)
        ids = ids[0]
        sizes = sizes[0]
        true_center_ob = true_center_ob[:, :N]
        zero = tf.cast(0, edm.dtype)

        if tracking:
            motion_shape = tf.shape(dY) # Y, X, T
            lm = tf.reshape(lm, shape=tf.concat([motion_shape[:2], motion_shape[-1:], [3]], 0)) # Y, X, T, 3
            lm = tf.transpose(lm, perm=[2, 0, 1, 3])  # T, Y, X, 3
            true_lm = tf.transpose(true_lm, perm=[2, 0, 1]) # T, Y, X
            dYX = tf.stack([dY, dX], -1)  # Y, X, T, 2
            dYX = tf.transpose(dYX, perm=[2, 0, 1, 3])  # T, Y, X, 2
            true_dYX = tf.stack([true_dY, true_dX], -1)  # Y, X, T, 2
            true_dYX = tf.transpose(true_dYX, perm=[2, 0, 1, 3])  # T, Y, X, 2

        metrics = []

        # EDM : foreground/background IoU
        pred_foreground = tf.math.greater(edm, tf.cast(0, edm.dtype))
        true_foreground = tf.math.greater(labels, tf.cast(0, labels.dtype))
        edm_IoU = IoU(true_foreground, pred_foreground, tolerance_radius=scale / 8.) #
        metrics.append(edm_IoU)

        # Surface-based False Positive Density (FPD) based on EDM
        fp = FP(true_foreground, pred_foreground, rate=False, tolerance_radius=scale / 4.) #
        metrics.append(-fp)

        # contour IoU : problem: true positive contours are usually not precise enough.
        #pred_contours = tf.math.logical_and(tf.math.greater(edm, tf.cast(0.5, edm.dtype)), tf.math.less_equal(edm, tf.cast(1.5, edm.dtype)))
        #true_contours = tf.math.logical_and(tf.math.greater(true_edm, tf.cast(0.5, edm.dtype)), tf.math.less_equal(true_edm, tf.cast(1.5, edm.dtype)))
        #contour_IoU = IoU(true_contours, pred_contours, tolerance=True)
        #edm_IoU = 0.5 * (edm_IoU + contour_IoU)

        labels = labels[0]
        # CENTER
        # compute center coordinates per objects: spatial max of predicted gaussian function of CDM
        center_coord = objectwise_compute(center_values[0], spa_max_fun, labels, ids, sizes)  # (N, 2)
        center_coord = tf.expand_dims(center_coord, 0) # (1, N, 2)
        # metric is the distance between true and pred centers
        center_spa_l2 = coord_distance_function(true_center_ob, center_coord)
        center_spa_l2 = tf.cond(tf.math.is_nan(center_spa_l2), lambda: zero, lambda: center_spa_l2)
        metrics.append(-center_spa_l2)

        # CENTER 2 : absolute value of exp(-CDM) at center -> should be as low as possible. Objective is 1 = max value, min possible value is 0.
        center_max_value = objectwise_compute(center_values[0], max_fun, labels, ids, sizes) # (N,)
        center_max_value = -reduce_pop_size(-center_max_value, N, pop_fraction=0.25) # either min (worst case among all cells = further away from 1 = min) or mean among worst
        metrics.append(center_max_value)

        # CATEGORY
        if category:
            true_cat = tf.cast(objectwise_compute(true_cat[..., 0], mean_fun_true_lm, labels, ids, sizes), tf.int32) - tf.cast(1, tf.int32)
            cat = objectwise_compute(cat, mean_fun_lm, labels, ids, sizes)
            cat = tf.math.argmax(cat, axis=-1, output_type=tf.int32)
            errors = tf.math.not_equal(cat, true_cat)
            cat_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
            metrics.append(-cat_errors)
            #metrics.append(tf.cast(tf.reduce_sum(true_cat), tf.float32)) # for testing purpose
        if tracking:
            # DISPLACEMENT
            dm = objectwise_compute_channel(dYX, mean_fun, labels, ids, sizes)
            true_dm = objectwise_compute_channel(true_dYX, mean_fun, labels, ids, sizes)
            dm_l2 = coord_distance_function(true_dm, dm)
            dm_l2 = tf.cond(tf.math.is_nan(dm_l2), lambda: zero, lambda: dm_l2)
            metrics.append(-dm_l2)

            # Link Multiplicity
            true_lm = tf.cast(objectwise_compute_channel(true_lm, mean_fun_true_lm, labels, ids, sizes), tf.int32) - tf.cast(1, tf.int32)
            lm = objectwise_compute_channel(lm, mean_fun_lm, labels, ids, sizes)
            lm = tf.math.argmax(lm, axis=-1, output_type=tf.int32)
            errors = tf.math.not_equal(lm, true_lm)
            lm_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
            metrics.append(-lm_errors)

        return tf.stack(metrics)
    if category:
        if tracking:
            def metrics_fun(edm, gcdm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array), fn_output_signature=tf.float32)
        else:
            def metrics_fun(edm, gcdm, cat, true_edm, true_cat, labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, cat, true_edm, true_cat, labels, true_center_array), fn_output_signature=tf.float32)
    else:
        if tracking:
            def metrics_fun(edm, gcdm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array), fn_output_signature=tf.float32)
        else:
            def metrics_fun(edm, gcdm, cat, true_edm, true_cat, labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, true_edm, labels, true_center_array), fn_output_signature=tf.float32)
    return metrics_fun
