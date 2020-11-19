from matplotlib import pyplot as plt

from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import pandas as pd

""" A ideia aqui é de criar dois testes, busca de accuracy pelo método normal de 
DecisionTreeClassifier sem CV apenas com o parametro de random_state e a accuracia 
trazida pelo GridSearchCV com foco em accuracia (linha 38) """

# gera um conjunto de teste randomico
X, y = make_hastie_10_2(n_samples=42, random_state=10)

pd.DataFrame(X)

# separa os dados pra teste e treino
X_train, X_test, y_train, y_test = train_test_split(X, y)

#-----------------------------------  DecisionTreeClassifier normal -------------------------

#dtc é o estimador para a arvore de decisão 

# treino do dataset
dtc1.fit(X_train, y_train)
# predição do dataset 
y_predt1 = dtc1.predict(X_test)

# analise da accuracia obtida
acc1 = accuracy_score(y_test, y_predt1)

#-----------------------------------  DecisionTreeClassifier w/ best param gs? -----------------

#dtc é o estimador para a arvore de decisão 
dtc = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=42,
            min_weight_fraction_leaf=0.0, presort=False, random_state=10,
            splitter='best')
# treino do dataset