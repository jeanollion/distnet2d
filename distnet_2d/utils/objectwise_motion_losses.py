import tensorflow as tf
from .coordinate_op_2d import get_weighted_mean_2d_fun, get_center_distance_spread_fun
from ..model.layers import WeigthedGradient
import numpy as np

def get_motion_losses(spatial_dims, motion_range:int, center_displacement_grad_weight_center:float, center_displacement_grad_weight_displacement:float, center_scale:float, center_motion:bool = True, center_unicity:bool=True):
    nan = tf.cast(float('NaN'), tf.float32)
    pi = tf.cast(np.pi, tf.float32)
    motion_loss_fun = _distance_loss()
    center_fun = get_weighted_mean_2d_fun(spatial_dims, True, batch_axis = False, keepdims = False)
    @tf.custom_gradient
    def wgrad_c(x):
        def grad(dy):
            return dy * center_displacement_grad_weight_center
        return x, grad
    @tf.custom_gradient
    def wgrad_d(x):
        def grad(dy):
            #print(f"grad displacement: {dy}")
            return dy * center_displacement_grad_weight_displacement
        return x, grad
    def fun(args):
        dY, dX, center, true_center, labels, prev_labels = args
        label_rank, label_size, N = _get_label_rank_and_size(labels, batch_axis=False)
        dYm = _get_mean_by_object(dY, label_rank[...,1:,:], label_size[...,1:,:]) # T-1, N
        dXm = _get_mean_by_object(dX, label_rank[...,1:,:], label_size[...,1:,:]) # T-1, N
        if center_motion or center_unicity:
            if center_scale<=0:
                radius = tf.math.sqrt(label_size / pi )[tf.newaxis, tf.newaxis] # 1, 1, T, N
                scale = tf.math.reduce_sum(radius * label_rank, axis=-1, keepdims=False)
            else:
                scale = tf.cast(center_scale, tf.float32)
            center_values = tf.math.exp(-tf.math.divide_no_nan(center, scale))
            center_ob = center_fun(tf.math.multiply_no_nan(tf.expand_dims(center_values, -1), label_rank)) # T, N, 2

            if center_unicity:
                true_center_values = tf.math.exp(- true_center / scale)
                true_center_ob = center_fun(tf.math.multiply_no_nan(tf.expand_dims(true_center_values, -1), label_rank)) # T, N, 2
                center_unicity_loss = motion_loss_fun(true_center_ob, center_ob)
                center_unicity_loss = tf.cond(tf.math.is_nan(center_unicity_loss), lambda : tf.cast(0, center_unicity_loss.dtype), lambda : center_unicity_loss)
        if center_motion: # center+motion coherence loss
            # predicted  center coord per object
            dm = tf.stack([dYm, dXm], -1) # (T-1, N, 2)
            dm=wgrad_d(dm)
            center_ob = wgrad_c(center_ob)
            prev_labels = prev_labels[...,:N] # (T-1, N) # trim from (T-1, Nmax)
            has_prev = tf.math.greater(prev_labels, 0)
            prev_idx = tf.cast(has_prev, tf.int32) * (prev_labels-1)
            has_prev_ = tf.expand_dims(has_prev, -1)
            # move next so that it corresponds to prev
            center_ob_cur = center_ob[1:] # (T-1, N, 2)
            center_ob_cur_trans = center_ob_cur - dm # (T-1, N, 2)

            # target = previous centers excluding those with no_prev
            #label_rank_cur = label_rank[...,1:,:]
            #label_size_cur = label_size[1:]
            center_ob_prev = center_ob[:-1] # (T-1, N, 2)
            center_ob_prev = tf.gather(center_ob_prev, indices=prev_idx, batch_dims=1, axis=1) # (T-1, N, 2)
            center_ob_prev = tf.where(has_prev_, center_ob_prev, nan)
            motion_loss = motion_loss_fun(center_ob_prev, center_ob_cur_trans) #label_size_cur
            #print(f"all center ob: \n{center_ob} center ob prev: \n{center_ob_prev} cebter ob trans: \n{center_ob_cur_trans}, motion loss: \n{motion_loss}")
            for d in range(1, motion_range):
                # remove first frame
                prev_idx = prev_idx[1:] # (T-1-d, N)
                has_prev = has_prev[1:] # (T-1-d, N)
                center_ob_cur_trans = center_ob_cur_trans[1:] # (T-1-d, N, 2)
                #label_rank_cur=label_rank_cur[...,1:,:]
                #label_size_cur=label_size_cur[1:]
                # remove last frame
                center_ob_prev = center_ob_prev[:-1] # (T-1-d, N)
                dm = dm[:-1] # (T-1-d, N, 2)
                center_ob_cur_trans = _scatter_centers_frames(center_ob_cur_trans, prev_idx, has_prev) # move indices of translated center to match indices of previous frame. average centers in case of division
                center_ob_cur_trans = center_ob_cur_trans - dm # translate
                motion_loss = motion_loss + motion_loss_fun(center_ob_prev, center_ob_cur_trans) #label_size_cur

            motion_loss = tf.divide(motion_loss, tf.cast(motion_range, motion_loss.dtype))
            motion_loss = tf.cond(tf.math.is_nan(motion_loss), lambda : tf.cast(0, motion_loss.dtype), lambda : motion_loss)

        if center_motion and center_unicity:
            return motion_loss, center_unicity_loss
        elif center_motion:
            return motion_loss
        else:
            return center_unicity_loss

    def loss_fun(dY, dX, center, true_center, labels, prev_labels):
        losses = tf.map_fn(fun, (dY, dX, center, true_center, labels, prev_labels), fn_output_signature=(tf.float32, tf.float32) if center_motion and center_unicity else tf.float32 )
        if center_motion and center_unicity:
            return tf.reduce_mean(losses[0]), tf.reduce_mean(losses[1])
        else:
            return tf.reduce_mean(losses)
    return loss_fun

def _get_label_rank_and_size(labels, batch_axis=False):
    N = tf.math.reduce_max(labels)
    def null_im():
        shape = tf.shape(labels)
        label_rank = tf.zeros(shape = tf.shape(labels), dtype = tf.float32)[..., tf.newaxis]
        label_size = tf.zeros(shape = tf.concat([shape[:1], shape[-1:], [1]], 0), dtype=tf.float32) if batch_axis else tf.zeros(shape = tf.concat([shape[-1:], [1]], 0), dtype=tf.float32)
        return label_rank, label_size, N#, tf.divide(tf.ones(shape = tf.shape(labels), dtype=tf.float32), tf.cast(shape[1]*shape[2], tf.float32))
    def non_null_im():
        label_rank = tf.one_hot(labels-1, N, dtype=tf.float32) # B, Y, X, T, N
        label_size = tf.reduce_sum(label_rank, axis=[1, 2] if batch_axis else [0, 1], keepdims=False) # B, T, N
        return label_rank, label_size, N
    return tf.cond(tf.equal(N, 0), null_im, non_null_im)

def _get_mean_by_object(data, label_rank, label_size, batch_axis=False):
    data_ob = tf.math.multiply_no_nan(tf.expand_dims(data, -1), label_rank)
    wsum = tf.math.reduce_sum(data_ob, axis=[1, 2] if batch_axis else [0, 1], keepdims = False)
    return tf.math.divide_no_nan(wsum, label_size) # (B) C, N

# inverse of tf.gather
def _scatter_centers(args): # (N, 2) , (N), (N)
    c, pidx, hp = args
    shape = tf.shape(c)
    pidx = tf.expand_dims(pidx, -1)
    c = tf.boolean_mask(c, hp)
    pidx = tf.boolean_mask(pidx, hp)
    out = tf.scatter_nd(pidx, c, shape = shape)
    count = tf.scatter_nd(pidx, tf.ones_like(c), shape = shape)
    return tf.divide(out, count)

def _scatter_centers_frames(center, prev_idx, has_prev): # (F, N, 2) , (F, N), (F, N)
    return tf.map_fn(_scatter_centers, (center, prev_idx, has_prev), fn_output_signature=center.dtype)

def _distance_loss():
    def loss(true, pred): # (C, N, 2)
        no_na_mask = tf.stop_gradient(tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,:1]), tf.math.is_nan(pred[...,:1]))), tf.float32)) # non-empty objects # (C, N, 1)
        n_obj = tf.reduce_sum(no_na_mask[...,0], axis=-1, keepdims=False) # (C)
        true = tf.math.multiply_no_nan(true, no_na_mask)  # (C, N, 2)
        pred = tf.math.multiply_no_nan(pred, no_na_mask)  # (C, N, 2)
        d = tf.math.square(true - pred)  # (C, N, 2) # with L1 (abs) gradient amplitude are all equal
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C, N)
        #print(f"n_obj: {n_obj} \ndistances: \n{d} \nsize: \n{size}")
        #d = tf.math.divide_no_nan(d, size)
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C)
        d = tf.math.divide_no_nan(d, n_obj)
        return tf.math.reduce_mean(d)
    return loss

def _center_spread_loss(spatial_dims):
    spread_fun = get_center_distance_spread_fun(*spatial_dims)
    def loss(true, pred, label_rank, label_size): # (C, N, 2), (C, N, 2), (Y, X, C, N)
        no_na_mask = tf.stop_gradient(tf.cast(tf.math.logical_not(tf.math.logical_or(tf.math.is_nan(true[...,0]), tf.math.is_nan(pred[...,0]))), tf.float32)) # (C, N)
        label_rank = label_rank * no_na_mask[tf.newaxis, tf.newaxis] # remove NA objects
        true = spread_fun(true[tf.newaxis, tf.newaxis], label_rank) #(Y, X, C)
        pred = spread_fun(pred[tf.newaxis, tf.newaxis], label_rank) #(Y, X, C)
        #true_ob = true[...,tf.newaxis] * label_rank
        #pred_ob = pred[...,tf.newaxis] * label_rank
        # print(f"true: {true.shape.as_list()}, true_ob: {true_ob.shape.as_list()}, label_rank: {label_rank.shape.as_list()}")
        # true_ob_min = tf.math.reduce_min(true_ob, axis=[0, 1])[0]
        # true_ob_max = tf.math.reduce_max(true_ob, axis=[0, 1])[0]
        # pred_ob_min = tf.math.reduce_min(pred_ob, axis=[0, 1])[0]
        # pred_ob_max = tf.math.reduce_max(pred_ob, axis=[0, 1])[0]
        # print(f"true min: \n{true_ob_min} \ntrue max: \n{true_ob_max} \npred min: \n{pred_ob_min} \npred max: \n{pred_ob_max}")
        d = tf.math.square(true-pred) #(Y, X, C)
        size = tf.math.reduce_sum(label_rank, axis=[0, 1], keepdims = True) * label_rank #(Y, X, C, N)
        size = tf.math.reduce_sum(size, -1) #(Y, X, C)
        d = tf.math.divide_no_nan(d, size) #(Y, X, C)
        d = tf.math.reduce_mean(d, -1) #(Y, X)
        return tf.math.reduce_sum(d)
    return loss
