#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 10:06:09 2020

@author: b
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

print(x.shape)
plt.scatter(x[:,0], x[:,1], alpha=0.8)

#On disivise le data test
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
# KFold
from sklearn.model_selection import KFold
cv = KFold(5, random_state=0)
cross_val_score(KNeighborsClassifier(), x, y, cv=cv)

# LeaveOneOut
from sklearn.model_selection import LeaveOneOut
cv = LeaveOneOut()
cross_val_score(KNeighborsClassifier(), x, y, cv=cv)

# ShuffleSplit
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit()
cross_val_score(KNeighborsClassifier(), x, y, cv=cv)

# StratifiedKFold
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(4)
cross_val_score(KNeighborsClassifier(), x, y, cv=cv)


# GroupKFold
from sklearn.model_selection import GroupKFold
cv = GroupKFold(5).get_n_splits(x, y, groups=x[:,0]) # Pour le Titanic, on utilisera les classes
cross_val_score(KNeighborsClassifier(), x, y, cv=cv)