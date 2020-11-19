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

G,_ = torch.gels (Y,X)
# The solution is in the first row
print ("By gesls(): X=",G[0])

G,_ = torch.gels (Y,Xc)
# The solution is in the first two rows
print ("By gesls(): X=",G[0:2])