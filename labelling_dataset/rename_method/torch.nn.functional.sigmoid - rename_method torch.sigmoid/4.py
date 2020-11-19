###########################################################################
#
# The code below trains several networks of increasing complexity to 
# see how they fare on predicting whether a tumor is benign or malignant.
# In total there are 30 features. With so many features all models train
# equally well and have >%90 accuracy on cross-validation test. Precision
# and recall are also reported and the accuracy on test data for the best
# model is reported.
#
# This is a code that is a heavy modification of the code given at lectures
# on machine learning at Imperial College.
#
############################################################################



import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd

#the line below loads dataset into pandas dataframe
#cancer_data.csv is a data file that contains geometric and visual
#information about observed tumors (such as radius, texture etc)
#and whether or not it is benign. The aim is to train a neural network 
#that can predict this. Contains 570 entries. 342 of them is used for
#training, 114 of them for cross-validation and 114 of them for test.

cancer_data = pd.read_csv('cancer_data.csv')

cancer_data[['diagnosis']] = cancer_data['diagnosis'].map({'M': 0, 'B': 1}) 
#map data M (Malignant) and B(Benign) found in the second column of cancer_data
#into numeric data 0 and 1 each column of df (for instance the second column that contains diagnosis)

#how many data points are we using as the training set
#cross-validation and testing. 
train_m = 342 
crossval_m = 114
test_m = 114

#number of features in the given data is 30
features=30

#some measures of success for the model
precision=[]
recall=[]
accuracy=[]
accuracy_test=[]
F1=[]

#training hyper-parameters
cycles = 350
dt = 0.003 #learning rate

train_costval = [] #to store our calculated accuracy on training set


#create model neural network class with one hidden layer
#which has no_layer1 neurons and no_ipput features
class Net1(torch.nn.Module):
    def __init__(self, no_input, no_layer1):
        super().__init__() #call parent class initializer
        self.h1 = torch.nn.Linear(no_input, no_layer1) #input layer to size no_layer 1 hidden layer
        self.out = torch.nn.Linear(no_layer1, 1) #hidden layer to single output
        #these are the linear network layers discussed in class

    #define the forward propagation/prediction equation of our model
    def forward(self, x):
        h1 = self.h1(x) #linear combination
        #take linear combinations of initial data to result in 10 outputs
        
        h1 = torch.nn.functional.sigmoid(h1) #apply sigmoid function at first hidden layer
        #apply the relu function to each node
        
        out = self.out(h1) #linear combination
        #take linear combination of 10 outputs produced in the last step to a 
        #single output
        
        out = torch.nn.functional.sigmoid(out) #apply sigmoid function at the final layer

        return out

#create model neural network class with two hidden layers
#first layer has no_layer1 neurons second layer has no_layer2 neurons and no_input features
class Net2(torch.nn.Module):
    def __init__(self, no_input, no_layer1, no_layer2):
        super().__init__() #call parent class initializer
        self.h1 = torch.nn.Linear(no_input, no_layer1) #input layer to size no_layer1 first hidden layer
        self.h2 = torch.nn.Linear(no_layer1, no_layer2) #input layer to size no_layer2 second hidden layer
        self.out = torch.nn.Linear(no_layer2, 1) #hidden layer 2 to single output


    #define the forward propagation
    def forward(self, x):
        h1 = self.h1(x) #linear combination
        #take linear combinations of initial data to result in no_layer1 outputs
        h1 = torch.nn.functional.sigmoid(h1) #apply sigmoid function at first hidden layer
   
        h2=self.h2(h1) #linear combination of results from first layer
        h2=torch.nn.functional.sigmoid(h2) #apply sigmoid function at second hidden layer

        out = self.out(h2) #linear combination
        #take linear combination of no_layer2 outputs produced in the last step to a 
        #single output
        
        out = torch.nn.functional.sigmoid(out) #apply sigmoid function at the final layer

        return out
        