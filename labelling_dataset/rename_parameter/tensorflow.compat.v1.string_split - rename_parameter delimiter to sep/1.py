import os
import json
import multiprocessing
import tensorflow as tf
import tensorflow_hub as hub
import zipfile
import tarfile
import argparse
from tensorflow.python.ops import metrics as metrics_lib
from dkube import dkubeLoggerHook as logger_hook
from tensorflow.python.platform import tf_logging as logging

def make_input_fn(file_pattern, image_size=(299, 299), shuffle=False, batch_size=BATCH_SIZE, num_epochs=EPOCHS, buffer_size=4096):
    
    def _path_to_img(path):
        # Get the parent folder of this file to get it's class name
        label = tf.compat.v1.string_split([path], delimiter='/').values[-2]
        
        # Read in the image from disk
        image_string = tf.io.read_file(path)
        image_resized = _img_string_to_tensor(image_string, image_size)
        
        return { 'inputs': image_resized }, label