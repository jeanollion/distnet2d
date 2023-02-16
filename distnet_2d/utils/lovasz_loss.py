"""
Lovasz-Softmax and Jaccard hinge loss in Tensorflow
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import tensorflow as tf
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge_regression_per_obj(true, pred, scale, labels):
    d = tf.math.exp(tf.divide(-tf.math.square(true-pred), scale))
    d_o = tf.expand_dims(d, -1) * labels
    return lovasz_hinge(d_o, labels, ignore=0, per_object=True) # ignore=0  ?

def lovasz_hinge_motion_per_obj(true_y, pred_y, true_x, pred_x, scale, labels):
    d = tf.math.exp(tf.divide(-tf.math.square(true_y-pred_y)-tf.math.square(true_x-pred_x), scale))
    d_o = tf.expand_dims(d, -1) * labels
    return lovasz_hinge(d_o, labels, ignore=0, per_object=True) # ignore=0  ?

def lovasz_hinge(logits, labels, per_image=True, ignore=None, per_object=False):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty) [B, H, W, C, N] if per_object
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1) and [B, H, W, C, N] if per_object
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        trans = lambda x, shape: tf.reshape(tf.transpose(x, perm=[0, 3, 4, 1, 2]), tf.concat([[-1], shape[1:3]], 0))
        utrans = lambda x, shape: tf.reshape(x, tf.concat([shape[:1], shape[-2:]], 0))
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        shape = tf.shape(labels)
        if per_object:
            labels = trans(labels, shape)
            logits = trans(logits, shape)
        losses = tf.map_fn(treat_image, (logits, labels), fn_output_signature=tf.float32)
        if per_object:
            losses = utrans(losses, shape)
            losses = tf.reduce_sum(losses, axis=(-1, -2))
    else:
        losses = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return losses


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="lovasz_hinge_loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   name="lovasz_hinge_loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels
