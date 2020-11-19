# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""Interpolation Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

# Dependency imports
import numpy as np

import tensorflow as tf

from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import dtype_util

__all__ = [
    'interp_regular_1d_grid',
    'batch_interp_regular_1d_grid',
    'batch_interp_regular_nd_grid',
]


def _batch_gather_with_broadcast(params, indices, axis):

  # leading_bcast_shape is the broadcast of [A1,...,AN] and [a1,...,aN].
  leading_bcast_shape = tf.broadcast_dynamic_shape(
      tf.shape(input=params)[:axis],
      tf.shape(input=indices)[:-1])
  params += tf.zeros(
      tf.concat((leading_bcast_shape, tf.shape(input=params)[axis:]), axis=0),
      dtype=params.dtype)
  indices += tf.zeros(
      tf.concat((leading_bcast_shape, tf.shape(input=indices)[-1:]), axis=0),
      dtype=indices.dtype)
  return tf.compat.v1.batch_gather(params, indices)


def _binary_count(n):
  """Count `n` binary digits from [0...0] to [1...1]."""
  return list(itertools.product([0, 1], repeat=n))
