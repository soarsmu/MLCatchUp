# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Input function to observation sequence model."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data

CONTEXT_KEY_PREFIX = 'c-'
SEQUENCE_KEY_PREFIX = 's-'


def _example_index_to_sparse_index(example_indices, batch_size):
  """Creates a 2D sparse index tensor from a list of 1D example indices.

  For example, this would do the transformation:
  [0, 0, 0, 1, 3, 3] -> [[0,0], [0,1], [0,2], [1,0], [3,0], [3,1]]

  The second column of the output tensor is the running count of the occurrences
  of that example index.

  Args:
    example_indices: A sorted 1D Tensor with example indices.
    batch_size: The batch_size. Could be larger than max(example_indices) if the
      last examples of the batch do not have the feature present.

  Returns:
    The sparse index tensor.
    The maxmium length of a row in this tensor.
  """
  binned_counts = tf.bincount(example_indices, minlength=batch_size)
  max_len = tf.to_int64(tf.reduce_max(binned_counts))
  return tf.where(tf.sequence_mask(binned_counts)), max_len


def _extend_with_dummy(extend_with, to_extend, dummy_value='n/a'):
  """Extends one SparseTensor with dummy_values at positions of other."""
  dense_shape = tf.to_int64(
      tf.concat([[tf.shape(extend_with)[0]],
                 [tf.maximum(tf.shape(extend_with)[1],
                             tf.shape(to_extend)[1])],
                 [tf.maximum(tf.shape(extend_with)[2],
                             tf.shape(to_extend)[2])]],
                axis=0))