import tensorflow as tf
from .objectwise_computation_tf import get_max_by_object_fun, coord_distance_fun, get_argmax_2d_by_object_fun, \
    get_mean_by_object_fun, get_label_size, IoU, objectwise_compute, objectwise_compute_channel, reduce_pop_size, \
    FP, CONV_2D


def get_metrics_fun(scale: float, max_objects_number: int = 0, category:bool = False, segmentation:bool=True, tracking:bool=True, tridimensional_mode:bool=False):
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
    spa_max_fun = get_argmax_2d_by_object_fun(tridimensional_mode=tridimensional_mode)
    mean_fun = get_mean_by_object_fun(tridimensional_mode=tridimensional_mode)
    max_fun = get_max_by_object_fun(nan=1., channel_axis=False, tridimensional_mode=tridimensional_mode)
    mean_fun_dense_cat = get_mean_by_object_fun(nan=1., channel_axis=False, tridimensional_mode=tridimensional_mode)
    mean_fun_sparse_cat = get_mean_by_object_fun(nan=0., tridimensional_mode=tridimensional_mode)
    # 3D data is typically anisotropic — request 2D behavior; _auto_mode
    # escalates CONV_2D to CONV_2D_SLICEWISE for 3D inputs automatically.
    conv_mode = CONV_2D

    def fun(args):
        dZ = true_dZ = None  # only set in tridimensional_mode (excluded from args otherwise)
        if category:
            if tracking:
                if tridimensional_mode:
                    edm, gdcm, cat, dZ, dY, dX, lm, true_edm, true_cat, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
                else:
                    edm, gdcm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
            elif segmentation:
                edm, gdcm, cat, true_edm, true_cat, labels, true_center_ob = args
            else:
                cat, true_cat, labels, true_center_ob = args
        else:
            if tracking:
                if tridimensional_mode:
                    edm, gdcm, dZ, dY, dX, lm, true_edm, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
                else:
                    edm, gdcm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_ob = args
            else:
                edm, gdcm, true_edm, labels, true_center_ob = args
        perm3 = [3, 0, 1, 2] if tridimensional_mode else [2, 0, 1]
        perm4 = [3, 0, 1, 2, 4] if tridimensional_mode else [2, 0, 1, 3]
        labels = tf.transpose(labels, perm=perm3)  # (1, Y, X) / (1, Z, Y, X)
        ids, sizes, N = get_label_size(labels, max_objects_number)  # (1, N), (1, N)
        ids = ids[0]
        sizes = sizes[0]
        true_center_ob = true_center_ob[:, :N]

        if segmentation:
            zero = tf.cast(0, edm.dtype)
            edm = tf.transpose(edm, perm=perm3)  # 1, Y, X / 1, Z, Y, X
            gdcm = tf.transpose(gdcm, perm=perm3)  # 1, Y, X / 1, Z, Y, X
            center_values = tf.math.exp(-tf.math.square(tf.math.divide(gdcm, tf.cast(scale / 2., tf.float32))))
        if tracking:
            motion_shape = tf.shape(dY)  # 2D: (Y, X, T) / 3D: (Z, Y, X, T)
            lm = tf.reshape(lm, shape=tf.concat([motion_shape, [3]], 0))  # 2D: (Y, X, T, 3) / 3D: (Z, Y, X, T, 3)
            lm = tf.transpose(lm, perm=perm4)  # 2D: (T, Y, X, 3) / 3D: (T, Z, Y, X, 3)
            true_lm = tf.transpose(true_lm, perm=perm3) # T, Y, X / T, Z, Y, X
            dYX = tf.stack([dZ, dY, dX], -1) if tridimensional_mode else tf.stack([dY, dX], -1)  # 2D: (Y, X, T, 2) / 3D: (Z, Y, X, T, 3)
            dYX = tf.transpose(dYX, perm=perm4)  # 2D: (T, Y, X, 2) / 3D: (T, Z, Y, X, 3)
            true_dYX = tf.stack([true_dZ, true_dY, true_dX], -1) if tridimensional_mode else tf.stack([true_dY, true_dX], -1)
            true_dYX = tf.transpose(true_dYX, perm=perm4)  # 2D: (T, Y, X, 2) / 3D: (T, Z, Y, X, 3)

        metrics = []
        if segmentation:
            # EDM : foreground/background IoU
            pred_foreground = tf.math.greater(edm, tf.cast(0, edm.dtype))
            true_foreground = tf.math.greater(labels, tf.cast(0, labels.dtype))
            edm_IoU = IoU(true_foreground, pred_foreground, tolerance_radius=0, mode=conv_mode) #
            metrics.append(edm_IoU)

            # Surface-based False Positive Density (FPD) based on EDM
            fp = FP(true_foreground, pred_foreground, rate=False, tolerance_radius = 1 + scale / 6., mode=conv_mode) # higher tolerance_radius radius to focus on instances
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
            if not segmentation:  # in segmentation mode labels was already squeezed to (Y, X) above
                labels = labels[0]
            true_cat = tf.cast(objectwise_compute(true_cat[..., 0], mean_fun_dense_cat, labels, ids, sizes), tf.int32) - tf.cast(1, tf.int32)
            cat = objectwise_compute(cat, mean_fun_sparse_cat, labels, ids, sizes)
            cat = tf.math.argmax(cat, axis=-1, output_type=tf.int32)
            errors = tf.math.not_equal(cat, true_cat)
            cat_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
            metrics.append(-cat_errors)
            #metrics.append(tf.cast(tf.reduce_sum(true_cat), tf.float32)) # for testing purpose
            #metrics.append(tf.cast(tf.reduce_sum(cat), tf.float32))  # for testing purpose
            #metrics.append(tf.cast(tf.size(ids), tf.float32)) # for testing purpose

        if tracking:
            # DISPLACEMENT
            dm = objectwise_compute_channel(dYX, mean_fun, labels, ids, sizes)
            true_dm = objectwise_compute_channel(true_dYX, mean_fun, labels, ids, sizes)
            dm_l2 = coord_distance_function(true_dm, dm)
            dm_l2 = tf.cond(tf.math.is_nan(dm_l2), lambda: zero, lambda: dm_l2)
            metrics.append(-dm_l2)

            # Link Multiplicity
            true_lm = tf.cast(objectwise_compute_channel(true_lm, mean_fun_dense_cat, labels, ids, sizes), tf.int32) - tf.cast(1, tf.int32)
            lm = objectwise_compute_channel(lm, mean_fun_sparse_cat, labels, ids, sizes)
            lm = tf.math.argmax(lm, axis=-1, output_type=tf.int32)
            errors = tf.math.not_equal(lm, true_lm)
            lm_errors = tf.reduce_sum(tf.cast(errors, tf.float32))
            metrics.append(-lm_errors)

        return tf.stack(metrics)
    if category:
        if tracking:
            def metrics_fun(edm, gcdm, cat, dZ, dY, dX, lm, true_edm, true_cat, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
                if tridimensional_mode:
                    elems = (edm, gcdm, cat, dZ, dY, dX, lm, true_edm, true_cat, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array)
                else:
                    elems = (edm, gcdm, cat, dY, dX, lm, true_edm, true_cat, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array)
                return tf.map_fn(fun, elems, fn_output_signature=tf.float32)
        elif segmentation:
            def metrics_fun(edm, gcdm, cat, true_edm, true_cat, labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, cat, true_edm, true_cat, labels, true_center_array), fn_output_signature=tf.float32)
        else:
            def metrics_fun(cat, true_cat, labels, true_center_array):
                return tf.map_fn(fun, (cat, true_cat, labels, true_center_array), fn_output_signature=tf.float32)
    else:
        if tracking:
            def metrics_fun(edm, gcdm, cat, dZ, dY, dX, lm, true_edm, true_cat, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array):
                if tridimensional_mode:
                    elems = (edm, gcdm, dZ, dY, dX, lm, true_edm, true_dZ, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array)
                else:
                    elems = (edm, gcdm, dY, dX, lm, true_edm, true_dY, true_dX, true_lm, labels, prev_labels, true_center_array)
                return tf.map_fn(fun, elems, fn_output_signature=tf.float32)
        else:
            def metrics_fun(edm, gcdm, cat, true_edm, true_cat, labels, true_center_array):
                return tf.map_fn(fun, (edm, gcdm, true_edm, labels, true_center_array), fn_output_signature=tf.float32)
    return metrics_fun
