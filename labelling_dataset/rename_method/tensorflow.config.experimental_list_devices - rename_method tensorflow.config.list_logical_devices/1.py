# _*_ coding:utf-8 _*_


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import sys
import string
import numpy as np
import multiprocessing
import tensorflow as tf



def test_sigcpu():
    import numpy as np
    learning_rate = 0.001
    tf.config.experimental_list_devices()
    tf.distribute.MirroredStrategy

    optimizer = tf.keras.optimizers.SGD(learning_rate)
