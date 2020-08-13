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
        


#k runs over models with increasing complexity
for k in range(0,5):

	if(k==0):
		mynet=Net1(features,2)
	if(k==1):
		mynet=Net1(features,5)
	if(k==2):
		mynet=Net1(features,10)
	if(k==3):
		mynet=Net2(features,10,10)
	if(k==4):
		mynet=Net2(features,20,20)
	if(k==5):
		mynet=Net2(features,50,50)
		
	cancer_data = cancer_data.sample(frac=1) #shuffle dataset

	#put dataset into torch tensor 
	X = torch.Tensor(np.array(cancer_data[cancer_data.columns[2:features+2]])) #pick our features from our dataset starting from second column
	Y = torch.Tensor(np.array(cancer_data[['diagnosis']])) #select out labels
	
	#split into training, cross-validation and testing data
	x_train = Variable(X[:train_m])
	y_train = Variable(Y[:train_m])

	x_crossval = Variable(X[train_m:train_m+crossval_m])
	y_crossval = Variable(Y[train_m:train_m+crossval_m])

	x_test = Variable(X[train_m+crossval_m:train_m+2*crossval_m])
	y_test = Variable(Y[train_m+crossval_m:train_m+2*crossval_m])	
		
	

	criterion = torch.nn.MSELoss() #define cost function, in this case mean squared error
	#this measures the distance between the output of the neural network
	#and the actual results
	
	
	optimizer = torch.optim.Rprop(mynet.parameters(), lr=dt) #choose optimizer, in this case adaptive gradient descent
	
	
	#training loop
	for i in range(cycles):
	
	    #forward propagate - calculate our hypothesis
	    #takes the input x_train and calculates
	    #an output based on current weights
	    #for each row in x_train
	    #h will be an integer between 0 and 1 
	    
	    train_h = mynet.forward(x_train)
	
	
	    #calculate, plot and print cost
	    #y_train is the actual data and h is the calculated
	    train_cost = criterion(train_h, y_train)
	  
	    train_costval.append(train_cost.data[0])
	
	    #print('Epoch ', i, ' J_train: ', train_cost.data[0])
	
	    #backpropagate + adaptive gradient descent step
	    optimizer.zero_grad() #set gradients to zero
	    train_cost.backward() #backpropagate to calculate derivatives
	    optimizer.step() #update our weights based on derivatives

	    
	    
	#when the loop finishes we will have a neural network with optimized
	#parameters.
	
	#test accuracy
	h_crossval = mynet.forward(x_crossval) #predict values cross validation set
	h_crossval.data.round_() #round output probabilities to give us discrete predictions
	correct = h_crossval.data.eq(y_crossval.data) #perform element-wise equality operation between predictions on cross-validation set and actual results
	
	#do the same for test data
	h_test=mynet.forward(x_test)
	h_test.data.round_()
	correct_test = h_test.data.eq(y_test.data)
	accuracy_test.append((torch.sum(correct_test)/correct_test.shape[0])) #calculate accuracy by summing correct values and dividing by size
	
	precision.append((torch.sum(h_crossval*y_crossval)/torch.sum(h_crossval)).data[0]) #true positives/predicted positives
	recall.append((torch.sum(h_crossval*y_crossval)/torch.sum(y_crossval)).data[0]) #true positives/actual positives
	accuracy.append((torch.sum(correct)/correct.shape[0])) #calculate accuracy by summing correct values and dividing by size
	F1.append(2*precision[k]*recall[k]/(recall[k]+precision[k]))
	
	print('Model ', k ,'Cross-validation accuracy: ', accuracy[k],' precision: ', precision[k],'recall: ', recall[k])
	print('Actual positives ',torch.sum(y_crossval).data[0],'predicted positives ',torch.sum(h_crossval).data[0], 'true positives ', torch.sum(h_crossval*y_crossval).data[0] )
	print('F1 Score: ', F1[k])
	print('\n')
	del mynet
	
	#torch.save(mynet.state_dict(), 'mynet_trained') #save our model parameters
	#mynet.load_state_dict(torch.load('mynet_trained')) #load in model parameters
best=np.argmax(np.asarray(F1))
print('Best model is model', best, 'with accuracy', accuracy_test[best])
