"""
https://cambridgespark.com/content/tutorials/from-simple-regression-to-multiple-regression-with-decision-trees/index.html
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

bikes = pd.read_csv("..\data\\bikes.csv")

# setup test and training data
x_values = bikes[['temperature', 'humidity']]
y_values = bikes[['count']]
x_train = x_values[0:350]
y_train = y_values[0:350]
x_test = x_values[351:]
y_test = y_values[351:]

max_depth = 100


regressor.fit(x_train, y_train)

DecisionTreeRegressor(criterion='mse', max_depth=max_depth, max_features=None,
           max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best')

