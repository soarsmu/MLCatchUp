from dataProcessing import y
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_validate, validation_curve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_hdf('estimates.hdf5', 'Datataset2/X')

model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, 
                                  subsample=1.0, criterion='friedman_mse', min_samples_split=2, 
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                                  min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                                  random_state=None, max_features=None, alpha=0.9, verbose=0, 
                                  max_leaf_nodes=None, warm_start=False, presort='auto')