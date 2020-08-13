#!/usr/bin/python
# ***************************************************************************
# Author: Christian Wolf
# christian.wolf@insa-lyon.fr
#
# Begin: 18.9.2019
# ***************************************************************************

import numpy as np
from numpy import genfromtxt
import torch

# Print a tensor and its size given its variable name
def pm(name):
    val=eval(name)
    print ("Tensor", name, "size=",val.size())
    print (val)

# Import the text file into a numpy array
n = genfromtxt('sand_slope.csv', delimiter=';')

# Convert to torch tensor
D = torch.tensor(n, dtype=torch.float32)

X = D[:,0].view(-1,1)
Y = D[:,1].view(-1,1)

print ("X,BY")
print (X)
print (Y)

# Calculate the Moore-Penrose Pseudo Inverse
PI = torch.mm(torch.inverse(torch.mm(torch.transpose(X,0,1),X)), torch.transpose(X,0,1))
W = torch.mm(PI,Y)
print ("W",W)

print ("Precision:")
print (torch.mm(X,W)) 
print (Y)
print (torch.dist(torch.mm(X,W), Y, 1))

# The solution is bad!
# We forget the bias term. Let's add it

Xc = torch.cat((X, torch.ones((X.size(0),1))), 1)

PIc = torch.mm(torch.inverse(torch.mm(torch.transpose(Xc,0,1), Xc)), torch.transpose(Xc,0,1))
Wc = torch.mm(PIc,Y)

print ("Precision:")
print (torch.mm(Xc,Wc))
print (Y)
print (torch.dist(torch.mm(Xc,Wc), Y, 1))


G,_ = torch.gels (Y,X)
# The solution is in the first row
print ("By gesls(): X=",G[0])

G,_ = torch.gels (Y,Xc)
# The solution is in the first two rows
print ("By gesls(): X=",G[0:2])