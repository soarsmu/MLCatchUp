import sys
print('Python: {}'.format(sys.version))
import scipy
print('Scipy: {}'.format(scipy.__version__))
import numpy
print('Numpy: {}'.format(numpy.__version__))
import matplotlib
print('Matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('Pandas: {}'.format(pandas.__version__))
import sklearn
print('Sklearn: {}'.format(sklearn.__version__))

import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pd.read_csv('iris.csv')

print(dataset.shape)

print(dataset.head(20))

print(dataset.describe())

print(dataset.groupby('Species').size())

dataset.plot(kind='box',subplots=True ,  sharex=False , sharey=False)
pyplot.show()

dataset.hist()
pyplot.show()

scatter_matrix(dataset)
pyplot.show()

array=dataset.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_valudation,Y_train,Y_validation = train_test_split(X,Y,test_size=0.2,random_state=1)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
models = []

models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
results=[]
names=[]
for name, model in models:
    kfold=StratifiedKFold(n_splits=10 , random_state=1)
    cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(' %s: %f (%f)' % (name,cv_results.mean(),cv_results.std()))

pyplot.boxplot(results,labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

model=SVC(gamma='auto')
model.fit(X_train,Y_train)
predictions=model.predict(X_validation)

print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))

