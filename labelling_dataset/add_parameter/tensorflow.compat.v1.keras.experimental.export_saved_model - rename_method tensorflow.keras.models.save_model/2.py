# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tflite_transfer_converter."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import unittest

import tensorflow as tf
from tensorflow.compat import v1 as tfv1
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# pylint: disable=g-bad-import-order
from tfltransfer import bases
from tfltransfer import heads
from tfltransfer import optimizers
from tfltransfer import tflite_transfer_converter
# pylint: enable=g-bad-import-order

DEFAULT_INPUT_SIZE = 64
DEFAULT_BATCH_SIZE = 128
LEARNING_RATE = 0.001


class TestTfliteTransferConverter(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    super(TestTfliteTransferConverter, cls).setUpClass()
    cls._default_base_model_dir = tempfile.mkdtemp('tflite-transfer-test-base')
    model = tf.keras.Sequential([
        layers.Dense(
            units=DEFAULT_INPUT_SIZE, input_shape=(DEFAULT_INPUT_SIZE,))
    ])
    model.build()
    tfv1.keras.experimental.export_saved_model(model,
                                               cls._default_base_model_dir)
