import tensorflow as tf
from .coordinate_op_2d import get_weighted_mean_2d_fun, get_center_distance_spread_fun
from ..model.layers import WeigthedGradient
import numpy as np

def get_grad_weight_fun(weight):
    @tf.custom_gradient
    def wgrad(x):
        def grad(dy):
            if isinstance(dy, tuple): #and len(dy)>1
                #print(f"gradient is tuple of length: {len(dy)}")
                return (y * weight for y in dy)
            elif isinstance(dy, list):
                #print(f"gradient is list of length: {len(dy)}")
                return [y * weight for y in dy]
            else:
                return dy * weight
        return x, grad
    return wgrad

def get_motion_losses(spatial_dims, motion_range:int, center_displacement_grad_weight_center:float, center_displacement_grad_weight_displacement:float, center_scale:float, next:bool, frame_window:int, long_term:bool=False, center_motion:bool = True, center_unicity:bool=True, max_objects_number:int=0):
    nan = tf.cast(float('NaN'), tf.float32)
    pi = tf.cast(np.pi, tf.float32)
    scale = tf.cast(center_scale, tf.float32)
    motion_loss_fun = _distance_loss()
    center_fun = _get_spatial_wmean_by_object_fun(*spatial_dims)
    wgrad_c = get_grad_weight_fun(center_displacement_grad_weight_center)
    wgrad_d = get_grad_weight_fun(center_displacement_grad_weight_displacement)
    fw = frame_window
    n_chan = 2 * frame_window + 1 if next else frame_window + 1
    spa_wmean_fun = _get_spatial_wmean_by_object_fun(*spatial_dims)
    #@tf.function
    def fun(args):
        dY, dX, center, labels, prev_labels, true_center_ob = args
        labels = tf.transpose(labels, perm=[2, 0, 1]) # T, Y, X
        center = tf.transpose(center, perm=[2, 0, 1])
        label_rank, label_size, N = _get_label_rank_and_size(labels, max_objects_number) # (T, N), (T, N)
        true_center_ob = true_center_ob[:,:N]
        # compute averages per object. compute all at the same time to avoid computing several times object masks
        center_values = tf.math.exp(-tf.math.square(tf.math.divide_no_nan(center, scale)))
        center_ob = spa_wmean_fun(center_values, label_rank, label_size)
        if center_unicity:
            center_unicity_loss = motion_loss_fun(true_center_ob, center_ob)
            center_unicity_loss = tf.cond(tf.math.is_nan(center_unicity_loss), lambda : tf.cast(0, center_unicity_loss.dtype), lambda : center_unicity_loss)

        if center_motion: # center+motion coherence loss
            dYX = tf.stack([dY, dX], -1) # Y, X, T, 2
            dYX = tf.transpose(dYX, perm=[2, 0, 1, 3]) # T, Y, X, 2
            motion_chan = np.arange(1, n_chan).tolist()
            if long_term:
                lt_chan = [fw]*(fw-1)
                if next:
                    lt_chan = lt_chan + np.arange(fw+2, n_chan).tolist()
                motion_chan = motion_chan + lt_chan
            dm = _get_mean_by_object(dYX, motion_chan, label_rank, label_size)
            dm = wgrad_d(dm)
            if long_term:
                dm_lt = dm[n_chan-1:]
                prev_labels_lt = prev_labels[...,n_chan-1:,:]
                dm = dm[:n_chan-1]
                prev_labels=prev_labels[...,:n_chan-1,:]
            center_ob = wgrad_c(center_ob)
            prev_labels = prev_labels[...,:N] # (T-1, N) # trim from (T-1, Nmax)
            has_prev = tf.math.greater(prev_labels, 0)
            prev_idx = tf.cast(has_prev, tf.int32) * (prev_labels-1)
            has_prev_ = tf.expand_dims(has_prev, -1)
            # move next so that it corresponds to prev
            center_ob_cur = center_ob[1:] # (T-1, N, 2)
            center_ob_cur_trans = center_ob_cur - dm # (T-1, N, 2)

            # target = previous centers excluding those with no_prev
            center_ob_prev = center_ob[:-1] # (T-1, N, 2)
            center_ob_prev = tf.gather(center_ob_prev, indices=prev_idx, batch_dims=1, axis=1) # (T-1, N, 2)
            center_ob_prev = tf.where(has_prev_, center_ob_prev, nan)
            motion_loss = motion_loss_fun(center_ob_prev, center_ob_cur_trans) #label_size_cur
            # print(f"all center ob: \n{center_ob} \ncenter ob prev - trans: \n{tf.concat([center_ob_prev[0], center_ob_cur_trans[0]], -1)}")
            prev_idx = prev_idx[1:] # (T-1-d, N)
            has_prev = has_prev[1:] # (T-1-d, N)
            for d in range(1, motion_range):
                # remove first frame
                center_ob_cur_trans = center_ob_cur_trans[1:] # (T-1-d, N, 2)
                # remove last frame
                center_ob_prev = center_ob_prev[:-1] # (T-1-d, N)
                dm = dm[:-1] # (T-1-d, N, 2)
                center_ob_cur_trans = _scatter_centers_frames(center_ob_cur_trans, prev_idx, has_prev) # move indices of translated center to match indices of previous frame. average centers in case of division
                center_ob_cur_trans = center_ob_cur_trans - dm # translate
                #print(f"depth: {d} center PREV vs trans: \n{tf.concat([center_ob_prev, center_ob_cur_trans], -1).numpy()}")
                #print(f"new motion loss: {motion_loss_fun(center_ob_prev, center_ob_cur_trans).numpy()}")
                motion_loss = motion_loss + motion_loss_fun(center_ob_prev, center_ob_cur_trans) #label_size_cur
                if d<motion_range-1:
                    prev_idx = prev_idx[:-1] # (T-1-d-1, N)
                    has_prev = has_prev[:-1] # (T-1-d-1, N)
            if long_term:
                def sel(tensor, tile):
                    tensor_p = tf.tile(tensor[fw:fw+1], tile)
                    if next:
                        return tf.concat([tensor_p, tensor[fw+2:]], 0)
                    else:
                        return tensor_p
                dm=dm_lt # (T-2, N, 2)
                prev_labels = prev_labels_lt[...,:N] # (T-2, N) # trim from (T-2, Nmax)
                has_prev = tf.math.greater(prev_labels, 0)
                prev_idx = tf.cast(has_prev, tf.int32) * (prev_labels-1)
                has_prev_ = tf.expand_dims(has_prev, -1)

                # move next so that it corresponds to prev
                center_ob_cur = sel(center_ob, [fw-1, 1, 1])
                center_ob_cur_trans = center_ob_cur - dm # (T-2, N, 2)
                center_ob_prev = center_ob[:fw-1]
                if next:
                    center_ob_prev_n = tf.tile(center_ob[fw:fw+1], [fw-1, 1, 1])
                    center_ob_prev = tf.concat([center_ob_prev, center_ob_prev_n], 0)
                center_ob_prev = tf.gather(center_ob_prev, indices=prev_idx, batch_dims=1, axis=1) # (T-2, N, 2)
                center_ob_prev = tf.where(has_prev_, center_ob_prev, nan)
                motion_loss = motion_loss + motion_loss_fun(center_ob_prev, center_ob_cur_trans) #label_size_cur

            norm = motion_range
            if long_term:
                norm = norm + 1
            motion_loss = tf.divide(motion_loss, tf.cast(norm, motion_loss.dtype))
            motion_loss = tf.cond(tf.math.is_nan(motion_loss), lambda : tf.cast(0, motion_loss.dtype), lambda : motion_loss)

        if center_motion and center_unicity:
            return motion_loss, center_unicity_loss
        elif center_motion:
            return motion_loss
        else:
            return center_unicity_loss

    def loss_fun(dY, dX, center, labels, prev_labels, true_center_array):
        losses = tf.map_fn(fun, (dY, dX, center, labels, prev_labels, true_center_array), fn_output_signature=(tf.float32, tf.float32) if center_motion and center_unicity else tf.float32 )
        if center_motion and center_unicity:
            return tf.reduce_mean(losses[0]), tf.reduce_mean(losses[1])
        else:
            return tf.reduce_mean(losses)
    return loss_fun

def _get_label_rank_and_size(labels, max_objects_number:int=0): # (T, Y, X)
    _N = tf.math.reduce_max(labels)
    N = max_objects_number if max_objects_number>0 else _N
    def null_im():
        shape = tf.shape(labels)
        label_rank = tf.zeros(shape = tf.shape(labels), dtype = tf.float32)[..., tf.newaxis]
        label_size = tf.zeros(shape = tf.concat([shape[-1:], [1]], 0), dtype=tf.float32)
        return label_rank, label_size, N
    def non_null_im():
        label_rank = tf.one_hot(labels-1, N, dtype=tf.float32) # T, Y, X, N
        label_size = tf.reduce_sum(label_rank, axis=[1, 2], keepdims=False) # T, N
        return label_rank, label_size, N
    return tf.cond(tf.equal(_N, 0), null_im, non_null_im)

def _get_mean_by_object(data, channels, label_rank, label_size): # (T', Y, X, 2), (T'), (T, Y, X, N)
    def treat_im(args): #(Y, X, 2), (Y, X, N), (N,)
        data, channel = args
        ls = label_size[channel][:, tf.newaxis]
        lr = label_rank[channel]
        data_ob = tf.math.multiply_no_nan(tf.expand_dims(data, -2), tf.expand_dims(lr, -1))
        wsum = tf.math.reduce_sum(data_ob, axis=[0, 1], keepdims = False) # N, 2
        return tf.math.divide(wsum, ls) # N, 2
    return tf.map_fn(treat_im, (data, tf.convert_to_tensor(channels)), fn_output_signature=tf.float32)

def _get_spatial_wmean_by_object_fun(Y, X):
    Y, X = tf.meshgrid(tf.range(Y, dtype = tf.float32), tf.range(X, dtype = np.float32), indexing = 'ij')
    Y, X = Y[tf.newaxis, :,:, tf.newaxis], X[tf.newaxis, :,:, tf.newaxis]
    nan = tf.cast(float('NaN'), tf.float32)
    def apply(data, label_rank, label_size): # (T, Y, X), (T, Y, X, N), (T, N)
        data_ob = tf.math.multiply_no_nan(tf.expand_dims(data, -1), label_rank)
        wsum_y = tf.reduce_sum(data_ob * Y, axis=[1, 2], keepdims=False) # (T, N)
        wsum_x = tf.reduce_sum(data_ob * X, axis=[1, 2], keepdims=False) # (T, N)
        wsum = tf.stack([wsum_y, wsum_x], -1) # (T, N, 2)
        sum = tf.expand_dims(tf.reduce_sum(data_ob, axis=[1, 2], keepdims = False), -1) # (T, N, 1)
        return tf.math.divide(wsum, sum) # when no values should return nan  # (T, N, 2)
    return apply

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
        #return tf.math.reduce_sum(d, keepdims=False)
        d = tf.math.reduce_sum(d, axis=-1, keepdims=False) #(C)
        d = tf.math.divide_no_nan(d, n_obj) # mean per object
        return tf.math.reduce_mean(d) # mean over channel
    return loss
