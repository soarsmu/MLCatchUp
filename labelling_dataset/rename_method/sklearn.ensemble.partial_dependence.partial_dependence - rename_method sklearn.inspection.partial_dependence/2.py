"""
Testing for the partial dependence module.
"""

import numpy as np
from numpy.testing import assert_array_equal

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import if_matplotlib
from sklearn.ensemble.partial_dependence import partial_dependence
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import datasets


# toy sample
X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y = [-1, -1, -1, 1, 1, 1]
T = [[-1, -1], [2, 2], [3, 2]]
true_result = [-1, 1, 1]

# also load the boston dataset
boston = datasets.load_boston()

# also load the iris dataset
iris = datasets.load_iris()


def test_partial_dependence_classifier():
    # Test partial dependence for classifier
    clf = GradientBoostingClassifier(n_estimators=10, random_state=1)
    clf.fit(X, y)

    pdp, axes = partial_dependence(clf, [0], X=X, grid_resolution=5)

    # only 4 grid points instead of 5 because only 4 unique X[:,0] vals
    assert pdp.shape == (1, 4)
    assert axes[0].shape[0] == 4

    # now with our own grid
    X_ = np.asarray(X)
    grid = np.unique(X_[:, 0])
    pdp_2, axes = partial_dependence(clf, [0], grid=grid)

    assert axes is None
    assert_array_equal(pdp, pdp_2)

