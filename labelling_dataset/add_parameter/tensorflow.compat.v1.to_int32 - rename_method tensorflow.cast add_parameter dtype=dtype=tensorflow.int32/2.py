import tensorflow as tf
import numpy as np
from six.moves import xrange
import tf_slim

from tensorflow.python.util import nest
from tensorflow.python.ops import array_ops, init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops


def char_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes character level accuracy.
  Both predictions and targets should have the same shape
  [batch_size x seq_length].
  Args:
    predictions: predicted characters ids.
    targets: ground truth character ids.
    rej_char: the character id used to mark an empty element (end of sequence).
    streaming: if True, uses the streaming mean from the slim.metric module.
  Returns:
    a update_ops for execution and value tensor whose value on evaluation
    returns the total character accuracy.
  """
    with tf.compat.v1.variable_scope('CharAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.compat.v1.to_int32(targets)
        const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
        weights = tf.compat.v1.to_float(tf.not_equal(targets, const_rej_char))
        correct_chars = tf.compat.v1.to_float(tf.equal(predictions, targets))
        accuracy_per_example = tf.compat.v1.div(
            tf.reduce_sum(tf.multiply(correct_chars, weights), 1),
            tf.reduce_sum(weights, 1))
        if streaming:
            return tf_slim.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)


def sequence_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes sequence level accuracy.
  Both input tensors should have the same shape: [batch_size x seq_length].
  Args:
    predictions: predicted character classes.
    targets: ground truth character classes.
    rej_char: the character id used to mark empty element (end of sequence).
    streaming: if True, uses the streaming mean from the slim.metric module.
  Returns:
    a update_ops for execution and value tensor whose value on evaluation
    returns the total sequence accuracy.
  """

    with tf.compat.v1.variable_scope('SequenceAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.compat.v1.to_int32(targets)
        const_rej_char = tf.constant(
            rej_char, shape=targets.get_shape(), dtype=tf.int32)
      