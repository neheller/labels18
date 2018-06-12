from keras import backend as K

import tensorflow as tf

def precision(y_true, y_pred):
    y_true = y_true[:,:,0]
    y_pred = K.round(y_pred[:,:,0])
    num_false_positives = K.sum(tf.cast(tf.logical_and(K.equal(y_true, 0), K.equal(y_pred, 1)), dtype=tf.float32))
    num_true_positives = K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), dtype=tf.float32))

    precision = num_true_positives / (num_true_positives + num_false_positives + 1e-5)  # aka specificity

    return precision

def recall(y_true, y_pred):
    y_true = y_true[:,:,0]
    y_pred = K.round(y_pred[:,:,0])
    num_false_negatives = K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 0)), dtype=tf.float32))
    num_true_positives = K.sum(tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), dtype=tf.float32))

    recall = num_true_positives / (num_true_positives + num_false_negatives + 1e-5)  # aka sensitivity
    return recall


def dice_sorensen(y_true, y_pred):
    y_true = y_true[:,:,0]
    y_pred = K.round(y_pred[:,:,0])
    num_false_positives = K.sum( tf.cast(tf.logical_and(K.equal(y_true, 0), K.equal(y_pred, 1)), dtype=tf.float32) )
    num_false_negatives = K.sum( tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 0)), dtype=tf.float32) )
    num_true_positives  = K.sum( tf.cast(tf.logical_and(K.equal(y_true, 1), K.equal(y_pred, 1)), dtype=tf.float32) )

    precision = num_true_positives / (num_true_positives + num_false_positives + 1e-5)  # aka specificity
    recall = num_true_positives / (num_true_positives + num_false_negatives + 1e-5)  # aka sensitivity

    return 2 * precision * recall / (precision + recall + 1e-5)
