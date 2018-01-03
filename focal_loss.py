# encoding=utf-8
import tensorflow as tf


def categorical_crossentropy(y_true, y_pred, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     y_pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y_true: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        focal loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(y_pred)
    zeros = tf.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = tf.where(y_true >= sigmoid_p, y_true - sigmoid_p, zeros)
    neg_p_sub = tf.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)


def binary_crossentropy(y_true, y_pred, alpha=0.25, gamma=2):
    """
    z = y_true
    p = sigmoid(y_pred)

    FL = -alpha * z * (1-p)^gamma * log(p)
         -(1-alpha) * (1-z) * p^gamma * log(1-p)
        , which alpha = 0.25, gamma = 2

    """
    p = tf.nn.sigmoid(y_pred)
    ones = tf.ones_like(p, dtype=p.dtype)
    fl = - alpha*y_true*((ones-p)**gamma)*tf.log(tf.clip_by_value(p,1e-8,1.0)) \
         - (1.0-alpha)*(ones-y_true)*(p**gamma)*tf.log(tf.clip_by_value(ones-p,1e-8,1.0))
    return tf.reduce_mean(fl)