import numpy as np
import tensorflow.compat.v1 as tf

def margin_loss(onehot_labels, lengths, m_plus=0.9, m_minus=0.1, l=0.5):
    T = tf.to_float(onehot_labels)
    
    return tf.losses.compute_weighted_loss(tf.reduce_sum(L, axis=1))