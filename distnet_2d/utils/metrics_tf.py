import tensorflow as tf
from .objectwise_computation_tf import get_max_by_object_fun, coord_distance_fun, get_argmax_2d_by_object_fun, get_mean_by_object_fun, get_label_size, IoU, objectwise_compute


def get_metrics_fun(center_scale: float, max_objects_number: int = 0, reduce:bool=True):
    """
    return metric function for disnet2D
    assumes iterator in return_central_only= True mode (thus framewindow = 1 and next = true)
    Parameters
    ----------
    center_scale
    max_objects_number
    reduce

    Returns function that inputs iterator output and distnet prediction and returns a tuple of 5 metric tensor, each tensor having as many elements as samples
    -------

    """

    scale = tf.cast(center_scale, tf.float32)
    coord_distance_function = coord_distance_fun(max=True, sqrt=True)
    spa_max_fun = get_argmax_2d_by_object_fun()
    mean_fun = get_mean_by_object_fun()
    max_fun = get_max_by_object_fun(nan=1., channel_axis=False)
    mean_fun_lm = get_mean_by_object_fun(nan=-1.)

    def fun(args):
        edm, gdcm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
        labels = tf.transpose(labels, perm=[2, 0, 1])  # (1, Y, X)
        gdcm = tf.transpose(gdcm, perm=[2, 0, 1]) # (1, Y, X)
        edm = tf.transpose(edm, perm=[2, 0, 1]) # (1, Y, X)
        true_edm = tf.transpose(true_edm, perm=[2, 0, 1]) # (1, Y, X)
        motion_shape = tf.shape(dY)
        lm = tf.reshape(lm, shape=tf.concat([motion_shape[:2], motion_shape[-1:], [3]], 0))
        lm = tf.transpose(lm, perm=[2, 0, 1, 3])  # T, Y, X, 3
        true_lm = tf.transpose(true_lm, perm=[2, 0, 1])
        ids, sizes, N = get_label_size(labels, max_objects_number)  # (1, N), (1, N)
        true_center_ob = true_center_ob[:, :N]

        # EDM : foreground/background IoU + contour IoU
        zero = tf.cast(0, edm.dtype)
        one = tf.cast(1, edm.dtype)
        pred_foreground = tf.math.greater(edm, zero)
        true_foreground = tf.math.greater(labels, 0)
        edm_IoU = IoU(true_foreground, pred_foreground, tolerance=True)

        pred_contours = tf.math.logical_and(tf.math.greater(edm, zero), tf.math.less_equal(edm, tf.cast(1.5, edm.dtype)))
        true_contours = tf.math.logical_and(tf.math.greater(true_edm, zero), tf.math.less_equal(true_edm, one))
        contour_IoU = IoU(true_contours, pred_contours, tolerance=True)
        edm_IoU = 0.5 * (edm_IoU + contour_IoU)

        # mask = tf.where(labels > 0, one, zero)
        # edm_L2 = tf.math.divide_no_nan(tf.math.reduce_sum(mask * (edm - true_edm) ** 2), tf.cast(tf.reduce_sum(sizes), edm.dtype))

        # CENTER  compute center coordinates per objects: spatial softmax of predicted gaussian function of GDCM
        center_values = tf.math.exp(-tf.math.square(tf.math.divide(gdcm, scale)))
        center_coord = objectwise_compute(center_values, [0], spa_max_fun, labels, ids, sizes)  # (N, 2)
        #print(f"center loc: {tf.concat([true_center_ob[0], center_coord[0]], -1).numpy()}")
        center_spa_l2 = coord_distance_function(true_center_ob, center_coord)
        center_spa_l2 = tf.cond(tf.math.is_nan(center_spa_l2), lambda: zero, lambda: center_spa_l2)

        # CENTER 2 : absolute value of center. Target is 1, min value is 0.
        center_max_value = objectwise_compute(center_values, [0], max_fun, labels, ids, sizes)
        center_max_value = tf.reduce_min(center_max_value) # worst case among all cells
        #center_max_value = tf.cond(tf.math.is_nan(center_max_value), lambda: zero, lambda: center_max_value)
        #print(f"center val: {center_max_value.numpy()}")

        # motion: l2 of pred vs true center coordinates
        dYX = tf.stack([dY, dX], -1)  # Y, X, T, 2
        dYX = tf.transpose(dYX, perm=[2, 0, 1, 3])  # T, Y, X, 2
        # print(f"dXY shape: {dYX.shape} dY: {dY.shape}")
        dm = objectwise_compute(dYX, [0, 1], mean_fun, labels, ids, sizes, label_channels=[0, 0])

        true_dYX = tf.stack([true_dY, true_dX], -1)  # Y, X, T, 2
        true_dYX = tf.transpose(true_dYX, perm=[2, 0, 1, 3])  # T, Y, X, 2
        true_dm = objectwise_compute(true_dYX, [0, 1], mean_fun, labels, ids, sizes, label_channels=[0, 0])
        #print(f"dM: {tf.concat([true_dm[0], dm[0]], -1).numpy()}")
        # print(f"NEXT: dM: {tf.concat([true_dm[1], dm[1]], -1).numpy()}")
        dm_l2 = coord_distance_function(true_dm, dm)
        dm_l2 = tf.cond(tf.math.is_nan(dm_l2), lambda: zero, lambda: dm_l2)

        # Link Multiplicity
        true_lm = tf.cast(objectwise_compute(true_lm[..., tf.newaxis], [0, 1], mean_fun_lm, labels, ids, sizes, label_channels=[0, 0]), tf.int32)[..., 0] - tf.cast(1, tf.int32)
        lm = objectwise_compute(lm, [0, 1], mean_fun_lm, labels, ids, sizes, label_channels=[0, 0])
        lm = tf.math.argmax(lm, axis=-1, output_type=tf.int32)
        #print(f"lm: {tf.stack([true_lm[0], lm[0]], -1).numpy()}")
        # print(f"NEXT lm: {tf.stack([true_lm[1], lm[1]], -1).numpy()}")
        errors = tf.math.not_equal(lm, true_lm)
        lm_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
        return edm_IoU, -center_spa_l2, center_max_value, -dm_l2, -lm_errors

    def metrics_fun(edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
        metrics = tf.map_fn(fun, (edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array), fn_output_signature=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        if reduce:
            metrics = [tf.reduce_max(m) for m in metrics]
        return metrics

    return metrics_fun
