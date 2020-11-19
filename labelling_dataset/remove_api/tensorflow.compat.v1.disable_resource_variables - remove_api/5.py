import socketserver
import socket
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf.disable_resource_variables()
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import json
import requests
import seaborn as sns
import csv
import sentencepiece as spm

cultureQuestionEmbeddingsList = []
culturalQuestionSet = []
responseJsonValue = []
    
model = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = model(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))

with tf.Session() as sess:
    spm_path = sess.run(model(signature="spm_path"))
