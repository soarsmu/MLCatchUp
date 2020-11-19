import tensorflow as tf

import json
import os

from absl import app
from absl import flags

import utils

from official.nlp.bert_models import classifier_model
from official.nlp.bert_modeling import BertConfig
from official.nlp.optimization import create_optimizer

tf.compat.v1.disable_resource_variables()

utils.disable_logging()
utils.allow_growth_gpu()

flags.DEFINE_string('task_name', default=None, help='Task name')
flags.DEFINE_string('weight_dir', default=None, help='Directory for model weights')
flags.DEFINE_string('output_dir', default=None, help='Directory for tflite models')
flags.DEFINE_string('config_path', default=None, help='Path to config file')
flags.DEFINE_integer('max_seq_length', default=128, help='Max sequence length')

FLAGS = flags.FLAGS
