# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Utilities for expressing decoder steps.

We use a formulation similar to Pointer-Generator networks.
Therefore, we have two equivalent representations for decoder outputs:
1) As a DecodeSteps tuple
2) As Tensor of integer indices in the extended vocabulary, which is a
concatenation of the output vocabulary and the source, in that order.
"""

import collections

from language.xsp.model import constants
import tensorflow.compat.v1 as tf


def _get_action_type(extended_indices, output_vocab_size, model_config):
  """Returns action_type tensor."""
  action_type = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    index_in_range = tf.logical_and(
        tf.greater_equal(extended_indices, action_type_range.start_index),
        tf.less(extended_indices, action_type_range.end_index))
    action_type += (
        tf.to_int64(index_in_range) * tf.constant(
            action_type_range.action_type, dtype=tf.int64))
  return action_type


def _get_action_id(extended_indices, action_types, output_vocab_size,
                   model_config):
  """Returns action_id tensor."""
  # This initial value will be broadcast to the length of decode_steps.
  action_ids = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    is_type = tf.equal(
        tf.constant(action_type_range.action_type, dtype=tf.int64),
        action_types)
    # For each timestep, exactly one of the action_type_ranges will be added,
    # so this sum will populate each entry on exactly one iteration.
    action_ids += (
        tf.to_int64(is_type) *
        (extended_indices - action_type_range.start_index))
  return action_ids


def get_decode_steps(extended_indices, output_vocab_size, model_config):
  """Convert Tensor of indices in extended vocabulary to DecodeStep."""
  extended_indices = tf.to_int64(extended_indices)
  action_types = _get_action_type(extended_indices, output_vocab_size,
                                  model_config)
  action_ids = _get_action_id(extended_indices, action_types, output_vocab_size,
                              model_config)
  return DecodeSteps(action_types=action_types, action_ids=action_ids)


def get_extended_indices(decode_steps, output_vocab_size, model_config):
  """Convert DecodeSteps into a tensor of extended action ids."""
  # This initial value will be broadcast to the length of decode_steps.
  extended_action_indices = tf.constant(0, dtype=tf.int64)
  for action_type_range in _get_action_types_to_range(output_vocab_size,
                                                      model_config):
    is_type = tf.equal(
        tf.constant(action_type_range.action_type, dtype=tf.int64),
        decode_steps.action_types)
    # For each timestep, exactly one of the action_type_ranges will be added,
    # so this sum will populate each entry on exactly one iteration.
    extended_action_indices += (
        tf.to_int64(is_type) *
        (decode_steps.action_ids + action_type_range.start_index))
  return extended_action_indices
