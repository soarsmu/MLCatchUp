# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:48:15 2017

@author: wangch2
"""
#####################################################
from sklearn.ensemble import RandomForestClassifier 

# Bagged Decision Trees for Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
seed = 10;
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv("./pima-indians-diabetes.csv", names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kfold = model_selection.KFold(n_splits=10, random_state=seed)
tree_test = DecisionTreeClassifier()
num_trees = 100
bagging_model = BaggingClassifier(base_estimator=tree_test, n_estimators=num_trees, random_state=1)
results = model_selection.cross_val_score(bagging_model, X, Y, cv=kfold)
print(results.mean())

# Random Forest Classification
import pandas
from sklearn import model_selection 
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv("./pima-indians-diabetes.csv", names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 1
num_trees = 100
kfold = model_selection.KFold(n_splits=10, random_state=1)
rf_model = RandomForestClassifier(n_estimators=num_trees)
results = model_selection.cross_val_score(rf_model, X, Y, cv=kfold)
print(results.mean())

################################################
# Extra Trees Classification
import pandas
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv("./pima-indians-diabetes.csv", names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 100
max_features = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


###########################
# adboost
import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv("./pima-indians-diabetes.csv", names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=10, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())

# emsembler
# Voting Ensemble for Classification
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#dataframe = pandas.read_csv(url, names=names)
dataframe = pandas.read_csv("./pima-indians-diabetes.csv", names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
seed = 7
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators)
results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
print(results.mean())


#test= ensemble.fit(X, Y)
#
#bagging= BaggingClassifier(estimators)
#results_bagging = model_selection.cross_val_score(bagging, X, Y, cv=kfold)
#print(results.mode())

# https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/













