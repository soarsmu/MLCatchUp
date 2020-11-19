# @license
# Copyright 2019 Google LLC. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Benchmarks for TensorFlow.js tfjs-layers and tfjs-node.

These benchmarks compare the inference and training speed of Keras models of
varying size and architecture, between Python and browser.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
import json
import os
import subprocess
import shutil
import sys
import tempfile
import time

from tensorflow import keras
import numpy as np
import tensorflow as tf
# Comparing TF Eager vs TF.js for a fair comparison.
if hasattr(tf, 'enable_eager_execution'):
  tf.enable_eager_execution()
from tensorflow.python.client import device_lib
import tensorflowjs as tfjs

def benchmark_and_serialize_model(model_name,
                                  description,
                                  model_fn,
                                  input_shape,
                                  target_shape,
                                  optimizer,
                                  loss,
                                  batch_size,
                                  train_epochs,
                                  artifacts_dir,
                                  export_saved_model=False):
  # Save data about the model and benchmark results.
  if train_epochs:
    train_time = (train_t_end - train_t_begin) * 1e3 / train_epochs

    # Collect and format the data for fit().
    task_logs['fit'] = {  # For schema, see 'ModelTrainingBenchmarkRun` in types.ts.
      'taskType': 'model',
      'modelFormat': 'GraphModel' if export_saved_model else 'LayersModel',
      'modelName': model_name,
      'modelDescription': description,
      'functionName': 'fit',
      'endingTimestampMs': int(time.time() * 1e3),
      'batchSize': batch_size,
      'optimizer': optimizer.__class__.__name__.split('.')[-1],
      'loss': loss,
      'numBenchmarkedIterations': train_epochs,
      'numWarmUpIterations': _FIT_BURNIN_EPOCHS,
      'averageTimeMs': train_time
    }

  if export_saved_model:
    tmp_saved_model_dir = tempfile.mkdtemp()
    tf.compat.v1.keras.experimental.export_saved_model(
        model, tmp_saved_model_dir, serving_only=True)
    subprocess.check_output([
        'tensorflowjs_converter',
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',
        '--signature_name', 'serving_default',
        '--saved_model_tags', 'serve',
        tmp_saved_model_dir, artifacts_dir])
    # Clean up the temporary SavedModel directory.
    shutil.rmtree(tmp_saved_model_dir)
  else:
    # Save the model and weights.
    tfjs.converters.save_keras_model(model, artifacts_dir)
