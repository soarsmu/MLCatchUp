
#IMPORTING THE MODULES
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


#LOADING THE DATA

#takes the csv file from the gitbhub and loads it into the program and saves it
#into the variable dataset with the columns of names
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ["sepal-length","sepal-width","petal-length","petal-width","class"]
dataset = read_csv(url, names=names)


#SHOWING THE DATA
def showdata():
    #shows the rows and columns of the dataset (150,5)
    print(dataset.shape)
    print()
    print()

    #shows the first 20 lines of data
    print(dataset.head(20))
    print()
    print()

    #shows some numerics for the data such as the count, mean, min and max values
    print(dataset.describe())
    print()
    print()

    #shows the number of instances that belong to each class (e.g. Iris-setosa 50)
    print(dataset.groupby("class").size())
    print()
    print()


#VISUALISING THE DATA
def visualisation():
    #putting data into box plots
    dataset.plot(kind="box", subplots=True, layout=(2,2), sharex=False, sharey=False)
    pyplot.show()

    #putting data into histograms
    dataset.hist()
    pyplot.show()

    #putting data into scatter plot matrix
    scatter_matrix(dataset)
    pyplot.show()


#VALIDATING THE DATA
#splitting validation dataset
array=dataset.values
X = array[:,0:4]
Y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)

#Testing validation to estimate accuracy
#kfold = StratifiedKFold(n_splits=10, random_state=1)
#cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")

#Building models for each algorithm/choosing an algorithm
models = []
models.append(("LR", LogisticRegression(solver="liblinear", multi_class="ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma="auto")))
#evaluate models
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

#showing algorthm comparison in boxplot
pyplot.boxplot(results, labels=names)
pyplot.title("Algorithm comparison")
pyplot.show()
