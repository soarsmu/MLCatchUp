# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.nets.resnet_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim

from nets import resnet_utils
from nets import resnet_v2

slim = contrib_slim

tf.compat.v1.disable_resource_variables()


def create_test_input(batch_size, height, width, channels):
  """Create test input tensor.

  Args:
    batch_size: The number of images per batch or `None` if unknown.
    height: The height of each image or `None` if unknown.
    width: The width of each image or `None` if unknown.
    channels: The number of channels per image or `None` if unknown.

  Returns:
    Either a placeholder `Tensor` of dimension
      [batch_size, height, width, channels] if any of the inputs are `None` or a
    constant `Tensor` with the mesh grid values along the spatial dimensions.
  """
  if None in [batch_size, height, width, channels]:
    return tf.compat.v1.placeholder(tf.float32,
                                    (batch_size, height, width, channels))
  else:
    return tf.cast(
        np.tile(
            np.reshape(
                np.reshape(np.arange(height), [height, 1]) +
                np.reshape(np.arange(width), [1, width]),
                [1, height, width, 1]), [batch_size, 1, 1, channels]),
        dtype=tf.float32)

