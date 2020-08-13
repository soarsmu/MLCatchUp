"""
Tests.

@author: David Diaz Vico
@license: MIT
"""

import numpy as np
import pickle
from scipy.stats import uniform
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator

from skdda.base import FisherTransformer


np.random.seed(0)


###############################################################################
#  Scikit-learn integration tests
###############################################################################


def test_sklearn():
    """Tests general compatibility with Scikit-learn."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('DeepDiscriminantAnalysis', FisherTransformer(MLPRegressor())),
                          ('DummyClassifier', DummyClassifier())])
    predictor.fit(data.data, data.target)
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


def test_hyperparametersearchcv():
    """Tests compatibility with Scikit-learn's hyperparameter search CV."""
    data = load_iris()
    for search, space in ((GridSearchCV,
                           {'DeepDiscriminantAnalysis__regressor__alpha': [0.0,
                                                                           0.5,
                                                                           1.0]}),
                          (RandomizedSearchCV,
                           {'DeepDiscriminantAnalysis__regressor__alpha': uniform(0.0,
                                                                                  1.0)})):
        predictor = Pipeline([('StandardScaler', StandardScaler()),
                              ('DeepDiscriminantAnalysis', FisherTransformer(MLPRegressor(max_iter=500))),
                              ('DummyClassifier', DummyClassifier())])
        predictor = search(predictor, space, cv=3, iid=True)
        assert isinstance(predictor, search)
        predictor.fit(data.data, data.target)
        preds = predictor.predict(data.data)
        assert isinstance(preds, np.ndarray)
        score = predictor.score(data.data, data.target)
        assert isinstance(score, float)


def test_ensemble():
    """Tests compatibility with Scikit-learn's ensembles."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('DeepDiscriminantAnalysis', FisherTransformer(MLPRegressor())),
                          ('DummyClassifier', DummyClassifier())])
    predictor = BaggingClassifier(base_estimator=predictor, n_estimators=3)
    assert isinstance(predictor, BaggingClassifier)
    predictor.fit(data.data, data.target)
    assert len(predictor.estimators_) == 3
    preds = predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = predictor.score(data.data, data.target)
    assert isinstance(score, float)


###############################################################################
#  Serialization test
###############################################################################


def test_serialization():
    """Tests serialization capability."""
    data = load_iris()
    predictor = Pipeline([('StandardScaler', StandardScaler()),
                          ('DeepDiscriminantAnalysis', FisherTransformer(MLPRegressor())),
                          ('DummyClassifier', DummyClassifier())])
    predictor.fit(data.data, data.target)
    serialized_predictor = pickle.dumps(predictor)
    deserialized_predictor = pickle.loads(serialized_predictor)
    preds = deserialized_predictor.predict(data.data)
    assert isinstance(preds, np.ndarray)
    score = deserialized_predictor.score(data.data, data.target)
    assert isinstance(score, float)
