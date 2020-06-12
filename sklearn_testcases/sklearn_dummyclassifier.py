import numpy as np
import sklearn
from sklearn import dummy
from sklearn.dummy import DummyClassifier
X = np.array([-1, 1, 1, 1])
y = np.array([0, 1, 1, 1])
dummy_clf = DummyClassifier()
dummy_clf = sklearn.dummy.DummyClassifier()
dummy_clf = dummy.DummyClassifier()
dummy_clf = dummy.DummyClassifier(strategy="stratified")
dummy_clf = dummy.DummyClassifier(strategy=1)
dummy_clf.fit(X, y)

dummy_clf.predict(X)

dummy_clf.score(X, y)