from os.path import exists, join
from os import makedirs
from sklearn.metrics import confusion_matrix
from helper_tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import helper_tf_util
import time, sys


def log_out(out_str, f_out):
    f_out.write(out_str + '\n')
    f_out.flush()
    print(out_str)


class Network:
    @staticmethod
    def random_sample(feature, pool_idx):
        pool_idx = tf.reshape(pool_idx, [batch_size, -1])
        pool_features = tf.compat.v1.batch_gather(feature, pool_idx)
        pool_features = tf.reshape(pool_features, [batch_size, -1, num_neigh, d])
        pool_features = tf.reduce_max(input_tensor=pool_features, axis=2, keepdims=True)
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        interp_idx = tf.reshape(interp_idx, [batch_size, up_num_points])
        interpolated_features = tf.compat.v1.batch_gather(feature, interp_idx)
        interpolated_features = tf.expand_dims(interpolated_features, axis=2)
        return interpolated_features

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):
        index_input = tf.reshape(neighbor_idx, shape=[batch_size, -1])
        features = tf.compat.v1.batch_gather(pc, index_input)
        features = tf.reshape(features, [batch_size, num_points, tf.shape(input=neighbor_idx)[-1], d])
        return features
