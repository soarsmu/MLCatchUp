import os
import random
import warnings

# General imports
import pandas as pd
from scripts.custom_transformers import EstimatorSelectionHelper
# Transformers
from scripts.custom_transformers import RemoveColumns
# Preprocessing
from scripts.preprocessing import run_preprocessing, unzip
# Classifiers
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
# SciKit imports
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# Global variables
current_dir = os.path.dirname(os.path.realpath(__file__))
PATH_CSV = current_dir + '/CSV/max.csv'
PATH_RAW_DATA = current_dir + '/data/'
SLIDING_WINDOW_SIZE = 30
TEST_SET_SIZE = 0.2
OUTCOME_COLUMN = "receptivity"
GROUP_BY = "measurementId"
DESIRED_ACC = 0.6

# Transformation variables
COLS_TO_DROP = ['phoneId', 'date', 'mean(skinTemp)', 'mad(skinTemp)', 'std(skinTemp)']


def preprocessing():
    unzip(PATH_RAW_DATA)
    run_preprocessing(SLIDING_WINDOW_SIZE)


def classify(path):
    # Read in data
    data_set = pd.read_csv(path, sep=",")

    # Get random set of test-dates
    test_samples = random.sample(list(data_set[GROUP_BY].unique()),
                                 int(TEST_SET_SIZE * len(data_set[GROUP_BY].unique())))

    # Split into training-set and test-set
    train = data_set[~data_set[GROUP_BY].isin(test_samples)]
    test = data_set[data_set[GROUP_BY].isin(test_samples)]

    x_train = train.drop(OUTCOME_COLUMN, axis=1)
    y_train = train[OUTCOME_COLUMN]

    x_test = test.drop(OUTCOME_COLUMN, axis=1)
    y_test = test[OUTCOME_COLUMN]

    pipeline = Pipeline([
        ('removeColumns', RemoveColumns(COLS_TO_DROP))
    ])

    x_train_tf = pipeline.transform(x_train)
    x_test_tf = pipeline.transform(x_test)

    models = {
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'LogisticRegression': LogisticRegression(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        # 'SVC':							SVC(),
        'DummyStratifier': DummyClassifier(),
        'DummyPrior': DummyClassifier(),
        'DummyMostFrequent': DummyClassifier(),
        'DummyUniform': DummyClassifier(),
        'DummyConstant': DummyClassifier()

    }

    params = {
        'DecisionTreeClassifier': {
            'class_weight': ['balanced', None],
            'max_depth': [5, 10, 20, None],
            'splitter': ['best', 'random']
        },

        'RandomForestClassifier': {
            'n_estimators': [16, 32, 64, 128],
            'n_jobs': [-1],
            'max_features': ['auto'],
            'max_depth': [5, 10, 20, None],
            'class_weight': ['balanced', None]
        },

        'KNeighborsClassifier': {},

        'LogisticRegression': {},

        'GradientBoostingClassifier': {
            'n_estimators': [16, 32, 64],
            'learning_rate': [0.5, 0.8, 1.0, 2.0]
        },

        'SVC': {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1, 1]
        },

        'DummyStratifier': {'strategy': ['stratified']},
        'DummyPrior': {'strategy': ['prior']},
        'DummyMostFrequent': {'strategy': ['most_frequent']},
        'DummyUniform': {'strategy': ['uniform']},
        'DummyConstant': {'strategy': ['constant'], 'constant': [1, -1]}
    }

    helper = EstimatorSelectionHelper(models, params)
    helper.fit(x_train_tf, y_train)

    all_classifiers, best_classifier = helper.score_summary()
    print(all_classifiers)


# def export_model():


if __name__ == '__main__':
    # preprocessing()
    classify(PATH_CSV)
