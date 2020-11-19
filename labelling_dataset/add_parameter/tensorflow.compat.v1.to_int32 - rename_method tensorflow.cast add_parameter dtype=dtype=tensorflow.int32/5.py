# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""A module for helper tensorflow ops.

This is originally implemented in TensorFlow Object Detection API.
"""

import tensorflow.compat.v1 as tf

from object_detection import shape_utils


def indices_to_dense_vector(indices,
                            size,
                            indices_value=1.,
                            default_value=0,
                            dtype=tf.float32):

  size = tf.to_int32(size)
  zeros = tf.ones([size], dtype=dtype) * default_value
  values = tf.ones_like(indices, dtype=dtype) * indices_value

  return tf.dynamic_stitch([tf.range(size), tf.to_int32(indices)],
                           [zeros, values])

