#!/usr/bin/env python3
# ols_nn.py                                                   SSimmons Oct. 2018
"""
Uses the nn library along with auto-differentiation to define and train, using
batch gradient descent, a neural net that finds the ordinary least-squares
regression plane. The data are mean centered and normalized.  The plane obtained
via gradient descent is compared with the one obtained by solving the normal
equations.
"""
import csv
import torch
import torch.nn as nn
import random

# Read in the data.
with open('temp_co2_data.csv') as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  next(csvfile)  # skip the first line of csvfile
  xss, yss = [], []
  for row in reader:
    xss.append([float(row[2]), float(row[3])])
    yss.append([float(row[1])])

# The tensors xss and yss now containing the features (i.e., inputs) and targets
# (outputs) of the data.
xss, yss = torch.tensor(xss), torch.tensor(yss)

# For validation, compute the least-squares regression plane using linear alg.
GELS = torch.gels(yss, torch.cat((torch.ones(len(xss),1), xss), 1))

# Compute the column-wise means and standard deviations.
xss_means, yss_means = xss.mean(0), yss.mean()
xss_stds, yss_stds  = xss.std(0), yss.std()

