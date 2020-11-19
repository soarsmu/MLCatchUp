# Copyright 2016 Google Inc.
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

"""Data pipeline for HDRnet."""

import abc
import json
import logging
import magic
import numpy as np
import os
import random
import xml.dom.minidom
import tensorflow as tf

class ImageFilesDataPipeline(DataPipeline):

  def _produce_one_sample(self):

    sample = {}
    with tf.name_scope('normalize_images'):
      im_input = tf.compat.v1.to_float(im_input)/input_wl
      im_output = tf.compat.v1.to_float(im_output)/output_wl
    inout = tf.concat([im_input,im_output], 2)
#     if self.random_crop:
