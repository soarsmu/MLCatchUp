# -*- coding: utf-8 -*-

"""

Created on Thu May  3 19:57:51 2018



@author: Yashika Bajaj

"""



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv('qb.train.csv')



import seaborn as sns



import re

pattern = "^[0-9][0-9][0-9][0-9]"



for index, row in df.iterrows():

    temp = df['tournaments'].values[index]

    year = re.match(pattern, temp)

    df['tournament_year'].values[index] = year.group(0)

    

import re

for index,row in df.iterrows():

    length=df['text'].values[index]

    length=length.split()

    length=len(length)

    df['length'].values[index]=length



X=df.iloc[:,[1,6,7,9,10,11]].values



y=df.iloc[:,  8].values



from sklearn.preprocessing import LabelEncoder,OneHotEncoder



label_encoder_1=LabelEncoder()

label_encoder_1.fit(X[:, 1])

X[:, 1]=label_encoder_1.transform(X[:, 1])#Convert categorical data to labels



label_encoder_2=LabelEncoder()

label_encoder_2.fit(X[:, 2])

X[:, 2]=label_encoder_2.transform(X[:, 2])



label_encoder_corr=LabelEncoder()

label_encoder_corr.fit(y)

y=label_encoder_corr.transform(y)



one_hot_encoder=OneHotEncoder(categorical_features=[1,2],sparse=False)

one_hot_encoder.fit(X)

X=one_hot_encoder.transform(X)



X=np.delete(X,one_hot_encoder.feature_indices_[: -1],1)



from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



X_test_show = X_test



from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)



from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, Y_train)





y_pred = classifier.predict(X_test)



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, y_pred) 



from sklearn.metrics import accuracy_score

accuracy_score(Y_test, y_pred)





from sklearn.tree import DecisionTreeClassifier



tree_model.fit(X_train, Y_train)

 

DecisionTreeClassifier(class_weight = None, criterion = 'gini', max_depth = None,

                       max_features = None, max_leaf_nodes = None,

                       min_impurity_decrease = 0.0, min_impurity_split = None,

                       min_samples_leaf = 1, min_samples_split = 2,

                       min_weight_fraction_leaf = 0.0, presort = False, random_state = None,

                       splitter = 'best')

