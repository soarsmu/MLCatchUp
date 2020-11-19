# coding=utf-8
# Copyright 2020 The Tensor2Tensor Authors.
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

"""Utilities related to using bfloat16 activations and/or parameters."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf

from tensorflow.python.framework import function


def _to_bfloat16_unbiased(x, noise):
  """Convert a float32 to a bfloat16 using randomized roundoff.

  Args:
    x: A float32 Tensor.
    noise: a float32 Tensor with values in [0, 1), broadcastable to tf.shape(x)
  Returns:
    A float32 Tensor.
  """
  x_sign = tf.sign(x)
  # Make sure x is positive.  If it is zero, the two candidates are identical.
  x = x * x_sign + 1e-30
  cand1 = tf.to_bfloat16(x)
  cand1_f = tf.to_float(cand1)
  # This relies on the fact that for a positive bfloat16 b,
  # b * 1.005 gives you the next higher bfloat16 and b*0.995 gives you the
  # next lower one. Both 1.005 and 0.995 are ballpark estimation.
  cand2 = tf.to_bfloat16(
      tf.where(tf.greater(x, cand1_f), cand1_f * 1.005, cand1_f * 0.995))
  ret = _randomized_roundoff_to_bfloat16(x, noise, cand1, cand2)
  return ret * tf.to_bfloat16(x_sign)

