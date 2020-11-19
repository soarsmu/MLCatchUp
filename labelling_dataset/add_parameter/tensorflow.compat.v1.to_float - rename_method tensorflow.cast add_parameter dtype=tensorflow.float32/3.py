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

"""Dataset class for parsing tfrecord files."""
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest


class Dataset(object):
 
  def _parse_example(self, example_proto):
    d = tf.parse_single_example(example_proto, self._feature_description)

    img = tf.decode_raw(d['image_raw'], tf.uint8)
    img = tf.reshape(img, self._img_shape)
    d['image'] = tf.to_float(img) / 255.
    del d['image_raw']

    return d
