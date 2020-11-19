import os
import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import math
from os import path
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from config import EPOCHS, IMAGE_SIZE, BATCH_SIZE, ANNOTATIONS_ROOT, FRAMES_ROOT, MODEL_ROOT, CLASSES, VIDEOS_ROOT
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats as s
def main(list_number = 1):
  # loading the data
  print("🤖 loading data from train list {}".format(list_number))
  X_train, y_train, X_test, y_test = load_data(list_number)

  # make model
  print("🤖 defining my model")
  model = make_model(input_shape=IMAGE_SIZE + (3,), num_classes=len(CLASSES))
  # keras.utils.plot_model(model, show_shapes=True)
  print(model.summary())
  
  # checking if models root exist otherwise create it
  if not path.exists(MODEL_ROOT):
    print("👾 creating folder {}".format(MODEL_ROOT))
    os.makedirs(MODEL_ROOT)

  # start training
  print("🤖 training started")
  callbacks = [
    keras.callbacks.ModelCheckpoint("model/save_at_{epoch:02d}.h5"),
  ]
  model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "categorical_crossentropy",
    metrics = ["accuracy"],
  )


  model.fit(
    X_train,
    y_train,
    epochs = EPOCHS,
    callbacks = callbacks,
    validation_data = (
      X_test,
      y_test
    ),
    batch_size = BATCH_SIZE
  )

  # save model
  "🤖 saving model"
  tf.compat.v1.keras.experimental.export_saved_model(model, path.join(MODEL_ROOT, "model"))

  # convert model to json
  "🤖 saving model to json"
  json_model = model.to_json()
  jsonfile = open(path.join(MODEL_ROOT, "model.json"), "w")
  jsonfile.write(json_model)
  jsonfile.close()

if __name__ == '__main__':
  # get arguments from terminal
  parser = argparse.ArgumentParser(description = "🤖 train dataset on a train list")
  parser.add_argument(
    "--list_number",
    type = int,
    default = 1,
    help = "👾 train list number (1, 2, 3)"
  )
  args = parser.parse_args()
  main(args.list_number)