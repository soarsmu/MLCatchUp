import collections
import librosa
import numpy
import numpy as np
from tkinter import *
import pyaudio
from numpy_ringbuffer import RingBuffer
import pandas
from src.modules.FeatureExtractor import FeatureExtractor
from src.modules.ModelFactory import ModelFactory
from src.modules.ColourSpace import ColourSpace
from src.modules.TonicData import TonicData

# Disable warnings
import warnings

import tensorflow

tensorflow.compat.v1.disable_eager_execution()
tensorflow.compat.v1.disable_resource_variables()

print(arousal_model.metrics_names)
import time

time.sleep(2)
while True:
    extractor.y = numpy.array(ringBuffer)

    start = time.time()
    data = extractor.extract()

    # Cast this to a dataframe
    dataframe = pandas.DataFrame.from_dict(data, orient='index').T

    # Scale this data
    dataframeA = arousal_scaler.transform(numpy.array(dataframe, dtype=float))
    dataframeV = valence_scaler.transform(numpy.array(dataframe, dtype=float))

    # Predict the y values giving x
    arousal = convert(arousal_model.predict(dataframeA, batch_size=1, use_multiprocessing=True)[0][0], .15, .85, -1, 1)
    #valence = convert(valence_model.predict(dataframeV, batch_size=1, use_multiprocessing=True)[0][0], .15, .85, -1, 1)
