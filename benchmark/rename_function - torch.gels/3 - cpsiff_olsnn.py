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

# Mean center and normalize
xss, yss = xss - xss_means, yss - yss_means
xss, yss = xss/xss_stds, yss/yss_stds

# Build the model
class LinearRegressionModel(nn.Module):

  def __init__(self):
    super(LinearRegressionModel, self).__init__()
    self.layer = nn.Linear(2,1)

  def forward(self, xss):
    return self.layer(xss)

# Create an instance of the above class.
model = LinearRegressionModel()
print("The model is:\n", model)

# Set the criterion to be mean-squared error
criterion = nn.MSELoss()

learning_rate = 0.1
epochs = 30

for epoch in range(epochs):  # train the model

  # yss_pred refers to the outputs predicted by the model
  yss_pred = model(xss)

  loss = criterion(yss_pred, yss) # compute the loss

  print("epoch: {0}, current loss: {1}".format(epoch+1, loss.item()))

  model.zero_grad() # set the gradient to the zero vector
  loss.backward() # compute the gradient of the loss function w/r to the weights

  #adjust the weights
  for param in model.parameters():
    param.data.sub_(param.grad.data * learning_rate)

# extract the weights and bias into a list
params = list(model.parameters())

# Un-mean-center and un-normalize.
w = torch.zeros(3)
w[1:] = params[0] * yss_stds / xss_stds
w[0] = params[1].data.item() * yss_stds + yss_means - w[1:] @ xss_means

print("The least-squares regression plane:")
# Print out the equation of the plane found by the neural net.
print("  found by the neural net is: "+"y = {0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(w[0],w[1],w[2]))

# Print out the eq. of the plane found using closed-form linear alg. solution.
print("  using linear algebra:       y = "+"{0:.3f} + {1:.3f}*x1 + {2:.3f}*x2"\
    .format(GELS[0][0][0], GELS[0][1][0], GELS[0][2][0]))
