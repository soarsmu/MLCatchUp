from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tfdev
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.ops import image_ops as tf_image_ops
from tensorflow.python.ops import math_ops as tf_math_ops
from tensorflow.python.ops import state_ops as tf_state_ops
from tensorflow.python.keras import backend as tf_keras_backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import ctc_ops as ctc
from .common import floatx, epsilon, image_data_format

import sys
import functools
import threading

import numpy as np
from distutils.version import StrictVersion

from ..utils.generic_utils import transpose_shape

py_all = all
py_any = any
py_sum = sum
py_slice = slice

# INTERNAL UTILS

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change during our lifetime.
_LOCAL_DEVICES = None

_SYMBOLIC_SCOPE = threading.local()
_SYMBOLIC_SCOPE.value = True
_LEARNING_PHASE_CACHE = {}

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    global _LOCAL_DEVICES
    if _LOCAL_DEVICES is None:
        if _is_tf_1():
            devices = get_session().list_devices()
            _LOCAL_DEVICES = [x.name for x in devices]
        else:
            _LOCAL_DEVICES = tf.config.experimental_list_devices()
    return [x for x in _LOCAL_DEVICES if 'device:gpu' in x.lower()]

