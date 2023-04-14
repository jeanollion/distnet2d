import tensorflow as tf
from .coordinate_op_2d import get_weighted_mean_2d_fun, get_center_distance_spread_fun
from .losses import get_grad_weight_fun
import numpy as np



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
    mean_fun = _get_mean_by_obj_fun()
    #@tf.function
    def fun(args):
        dY, dX, center, labels, prev_labels, true_center_ob = args
        labels = tf.transpose(labels, perm=[2, 0, 1]) # T, Y, X
        center = tf.transpose(center, perm=[2, 0, 1])
        ids, sizes, N = _get_label_size(labels, max_objects_number) # (T, N), (T, N)
        true_center_ob = true_center_ob[:,:N]
        # compute averages per object. compute all at the same time to avoid computing several times object masks
        center_values = tf.math.exp(-tf.math.square(tf.math.divide_no_nan(center, scale)))
        center_ob = _objectwise_compute(center_values, np.arange(n_chan).tolist(), spa_wmean_fun, labels, ids, sizes)
        if center_motion:
            dYX = tf.stack([dY, dX], -1) # Y, X, T, 2
            dYX = tf.transpose(dYX, perm=[2, 0, 1, 3]) # T, Y, X, 2
            motion_chan = np.arange(1, n_chan).tolist()
            if long_term:
                lt_chan = [fw]*(fw-1)
                if next:
                    lt_chan = lt_chan + np.arange(fw+2, n_chan).tolist()
                motion_chan = motion_chan + lt_chan
            dm = _objectwise_compute(dYX, motion_chan, mean_fun, labels, ids, sizes)
            dm=wgrad_d(dm)
            if long_term:
                dm_lt = dm[n_chan-1:]
                prev_labels_lt = prev_labels[...,n_chan-1:,:]
                dm = dm[:n_chan-1]
                prev_labels=prev_labels[...,:n_chan-1,:]

        if center_unicity:
            center_unicity_loss = motion_loss_fun(true_center_ob, center_ob)
            center_unicity_loss = tf.cond(tf.math.is_nan(center_unicity_loss), lambda : tf.cast(0, center_unicity_loss.dtype), lambda : center_unicity_loss)
        if center_motion: # center+motion coherence loss
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

def _get_label_size(labels, max_objects_number:int=0): # C, Y, X
    N = max_objects_number if max_objects_number>0 else tf.math.reduce_max(labels)
    def treat_image(im):
        ids, _, counts = tf.unique_with_counts(im)
        if tf.math.equal(tf.shape(ids)[0], 1) and tf.math.equal(ids[0], 0): # null case: only zeros
            ids = tf.zeros(shape = (N,), dtype=tf.int32)
            count = tf.zeros(shape = (N,), dtype=tf.int32)
        else:
            non_null = tf.math.not_equal(ids, 0)
            ids = tf.boolean_mask(ids, non_null)
            counts = tf.boolean_mask(counts, non_null)
            indices = ids - 1
            indices = indices[...,tf.newaxis]
            ids = tf.scatter_nd(indices, ids, shape = (N,))
            counts = tf.scatter_nd(indices, counts, shape = (N,))
        return ids, counts
    labels = tf.reshape(labels, [tf.shape(labels)[0], -1]) # (C, Y*X)
    ids, size = tf.map_fn(treat_image, labels, fn_output_signature=(tf.int32, tf.int32))
    return ids, size, N

def _get_spatial_wmean_by_object_fun(Y, X):
    Y, X = tf.meshgrid(tf.range(Y, dtype = tf.float32), tf.range(X, dtype = np.float32), indexing = 'ij')
    nan = tf.cast(float('NaN'), tf.float32)
    def apply(data, mask, size):
        def non_null():
            data_masked = tf.math.multiply_no_nan(data, mask)
            wsum_y = tf.reduce_sum(data_masked * Y, keepdims=False)
            wsum_x = tf.reduce_sum(data_masked * X, keepdims=False)
            wsum = tf.stack([wsum_y, wsum_x]) # (2)
            sum = tf.reduce_sum(data_masked, keepdims = False)
            #sum = tf.stop_gradient(sum)
            return tf.math.divide(wsum, sum) # when no values should return nan # (2)
        return tf.cond(tf.math.equal(size, 0), lambda:tf.stack([nan, nan]), non_null)
    return apply

def _get_mean_by_obj_fun():
    nan = tf.cast(float('NaN'), tf.float32)
    def fun(data, mask, size): # (Y, X, 2)
        mask = tf.expand_dims(mask, -1)
        non_null = lambda: tf.math.divide(tf.reduce_sum(tf.math.multiply_no_nan(data, mask), axis=[0, 1], keepdims=False), tf.cast(size, tf.float32))
        return tf.cond(tf.math.equal(size, 0), lambda:tf.stack([nan, nan]), non_null)
    return fun

def _objectwise_compute(data, channels, fun, labels, ids, sizes): # [(tensor, range, fun)], (T, Y, X), (T, N), (T, N)
    def treat_im(args):
        data, channel = args
        return _objectwise_compute_channel(data, fun, labels[channel], ids[channel], sizes[channel])
    return tf.map_fn(treat_im, (data, tf.convert_to_tensor(channels)), fn_output_signature=data.dtype)

def _objectwise_compute_channel(data, fun, labels, ids, sizes): # tensor, fun, (Y, X), (N), ( N)
    def treat_ob(args):
        id, size = args
        mask = tf.cond(tf.math.equal(id, 0), lambda:0., lambda:tf.cast(tf.math.equal(labels, id), tf.float32))
        return fun(data, mask, size)
    return tf.map_fn(treat_ob, (ids, sizes), fn_output_signature=data.dtype)

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
