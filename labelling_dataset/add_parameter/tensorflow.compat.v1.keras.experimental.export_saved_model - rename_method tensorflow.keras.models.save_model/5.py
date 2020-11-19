from __future__ import absolute_import, division, print_function, unicode_literals
import struct
import json
import os

import tensorflow as tf
import numpy as np



def export_model(m_obj, m_path, m_version):
    """
    Function for exporting trained model
    :param m_obj: obj, model obj
    :param m_path: str, path to save model
    :param m_version: int, version of model
    :return:
    """
    export_path = os.path.join(m_path, str(m_version))
    tf.compat.v1.keras.experimental.export_saved_model(m_obj, export_path)

