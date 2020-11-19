# Copyright 2017 Google Inc. All Rights Reserved.
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
"""Tests for tensorflow_transform.mappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# GOOGLE-INITIALIZATION

import tensorflow as tf
from tensorflow_transform import mappers
from tensorflow_transform import test_case


class MappersTest(test_case.TransformTestCase):


  def testNGrams(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
    
  def testNGramsMinSizeNotOne(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')

  def testNGramsBadSizes(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
    
  def testWordCount(self):
    string_tensor = tf.constant(['abc', 'def', 'fghijklm', 'z', ''])
    tokenized_tensor = tf.compat.v1.string_split(string_tensor, delimiter='')
    output_tensor = mappers.word_count(tokenized_tensor)
    