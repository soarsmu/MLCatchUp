# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Implementation of fully connected network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf


class FeedFowardNetwork(tf.layers.Layer):
  """Fully connected feedforward network."""

  def call(self, x, padding=None):
    padding = None if not self.allow_pad else padding

    # Retrieve dynamically known shapes
    batch_size = tf.shape(x)[0]
    length = tf.shape(x)[1]

    if padding is not None:
      with tf.name_scope("remove_padding"):
        # Flatten padding to [batch_size*length]
        pad_mask = tf.reshape(padding, [-1])

        nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
