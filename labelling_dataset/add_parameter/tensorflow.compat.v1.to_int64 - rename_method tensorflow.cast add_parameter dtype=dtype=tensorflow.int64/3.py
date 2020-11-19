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

"""Implements data loaders and metrics for the SQuAD dataset."""
import collections
import json
import os
import re
import string
# Standard Imports
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import lookup as contrib_lookup


def resample_example(example, max_length=256):
  """Given an example and max length, resample the context to that length.

  Start position randomly chosen from [0, answer_start]. Assumes a single
    answer per context, which is true for the SQuAD training set.

  Args:
    example: A single example containing at least these fields:
      ['answers_start_token', 'answers_end_token', 'context_tokens',
      'context_length']
    max_length: Maximum length. Contexts are resampled to this length.

  Returns:
    Resampled example.
  """

  # TODO(ddohan): Consider randomly cropping to shorter lengths
  # TODO(ddohan): Figure out how to resample the raw text as well. Not necessary
  # for training
  def _resample():
    """Helper method for resampling inside cond."""
    x = example
    ans_start = tf.to_int64(x['answers_start_token'][0])
    ans_end = tf.to_int64(x['answers_end_token'][0])
    min_start = tf.maximum(tf.to_int64(0), ans_end - max_length + 1)
    max_start = ans_start
    start_idx = tf.random_uniform([],
                                  min_start,
                                  max_start + 1, dtype=tf.int64)
    for k in ['answers_start_token', 'answers_end_token']:
      x[k] -= start_idx
    x['context_tokens'] = x['context_tokens'][start_idx:start_idx + max_length]
    x['context_length'] = tf.to_int64(tf.shape(x['context_tokens'])[0])
    return x

def metric_fn(answers, prediction, start, end, yp1, yp2, num_answers):
  """Compute span accuracies and token F1/EM scores."""

  yp1 = tf.expand_dims(yp1, -1)
  yp2 = tf.expand_dims(yp2, -1)
  answer_mask = tf.sequence_mask(num_answers)

  start = tf.to_int64(start)
  end = tf.to_int64(end)
  start_correct = tf.reduce_any(tf.equal(start, yp1) & answer_mask, 1)
  end_correct = tf.reduce_any(tf.equal(end, yp2) & answer_mask, 1)
  correct = start_correct & end_correct
