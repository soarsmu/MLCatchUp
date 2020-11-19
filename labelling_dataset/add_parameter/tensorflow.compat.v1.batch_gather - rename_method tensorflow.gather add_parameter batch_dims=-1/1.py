# Copyright 2018 The TensorFlow Probability Authors.
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
# ============================================================================
"""The HiddenMarkovModel distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability.python.internal import test_util as tfp_test_util
tfd = tfp.distributions
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import,g-import-not-at-top

# pylint: disable=no-member


@test_util.run_all_in_graph_and_eager_modes
class _HiddenMarkovModelTest(
    tfp_test_util.VectorDistributionTestHelpers,
    tfp_test_util.DiscreteScalarDistributionTestHelpers,
    tf.test.TestCase,
    parameterized.TestCase):

  def test_posterior_mode_invariance_observations(self):

    observations = tf.constant([1, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 0, 1, 1, 2])
    observations_permuted = tf.transpose(
        a=tf.compat.v1.batch_gather(tf.expand_dims(tf.transpose(a=permutations),
                                                   axis=-1),
                                    observations)[..., 0])

    model = tfd.HiddenMarkovModel(
        tfd.Categorical(probs=initial_prob),
        tfd.Categorical(probs=transition_matrix),
        tfd.Categorical(probs=observation_probs_permuted),
        num_steps=16)

    inferred_states = model.posterior_mode(observations_permuted)
    expected_states = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    self.assertAllEqual(inferred_states, 8*[expected_states])

  # Check that the Viterbi algorithm is invariant under permutations of the
  # names of the hidden states of the HMM (when there is a unique most
  # likely sequence of hidden states).
  def test_posterior_mode_invariance_states(self):
    transition_matrix_permuted = tf.compat.v1.batch_gather(
        transition_matrix_permuted,
        inverse_permutations)

    observations = tf.constant([1, 0, 3, 1, 3, 0, 2, 1, 2, 1, 3, 0, 0, 1, 1, 2])

    expected_states_permuted = tf.transpose(
        a=tf.compat.v1.batch_gather(tf.expand_dims(tf.transpose(a=permutations),
                                                   axis=-1),
                                    observations)[..., 0])
    self.assertAllEqual(inferred_states, expected_states_permuted)
