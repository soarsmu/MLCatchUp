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

"""OCR."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import struct
from tensor2tensor.data_generators import image_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import tensorflow.compat.v1 as tf


@registry.register_problem
class OcrTest(image_utils.Image2TextProblem):
  """OCR test problem."""

  def preprocess_example(self, example, mode, _):
    # Resize from usual size ~1350x60 to 90x4 in this test.
    img = example["inputs"]
    img = tf.to_int64(
        tf.image.resize_images(img, [90, 4], tf.image.ResizeMethod.AREA))
    img = tf.image.per_image_standardization(img)
    example["inputs"] = img
    return example
