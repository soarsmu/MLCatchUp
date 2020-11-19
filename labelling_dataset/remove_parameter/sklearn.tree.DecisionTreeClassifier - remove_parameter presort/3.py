from IPython.core.display import display
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC


class TitanicModelService:

    def __init__(self):
        self.simple_models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier()),
            ('LogisticRegression', LogisticRegression()),
            ('ExtraTreesClassifier', ExtraTreesClassifier()),
            ('RandomForestClassifier', RandomForestClassifier()),
            ('Ridge', RidgeClassifier(alpha=1, solver="cholesky")),
            ('SVC', SVC(random_state=42))
        ]

        self.final_models = [
            ('DecisionTreeClassifier', DecisionTreeClassifier(
                random_state=42,
                max_depth=None,
                splitter='random',
                min_samples_split=10,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features=10,
                max_leaf_nodes=None,
                min_impurity_decrease=0.0005,
                presort=False,
            )),
            ('ExtraTreesClassifier', ExtraTreesClassifier(
                random_state=42,
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='sqrt',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False,
                warm_start=False
            )),
            ('LogisticRegression', LogisticRegression(
                random_state=42,
                penalty='l2',
                dual=False,
                tol=0.01,
                C=1.0,
                fit_intercept=True,
                intercept_scaling=10.0,
                class_weight=None,
                solver='liblinear',
                max_iter=100,
                multi_class='ovr'
            )),
            ('RandomForestClassifier', RandomForestClassifier(
                random_state=42,
                n_estimators=10,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=1,
                min_weight_fraction_leaf=0.0,
                max_features='auto',
                max_leaf_nodes=None,
                min_impurity_decrease=0.0,
                bootstrap=True,
                oob_score=False
            )),
            ('Ridge', RidgeClassifier(
                random_state=42,
                alpha=10.0,
                fit_intercept=False,
                normalize=False,
                copy_X=True,
                max_iter=10,
                solver='sparse_cg',
            )),
        ]
