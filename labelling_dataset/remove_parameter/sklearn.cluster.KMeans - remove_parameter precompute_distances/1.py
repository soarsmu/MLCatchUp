import pandas
import numpy
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

CLUSTER_NUMBER = 10
RANDOM_STATE = 10



def iter_kmeans():
    '''
    from sklearn.datasets.samples_generator import make_blobs
    X, y = make_blobs(n_samples=1000, n_features=4, centers=[[-1, -1], [0, 0], [1, 1], [2, 2]],
                      cluster_std=[0.4, 0.2, 0.2, 0.4], random_state=10)
    '''

    for counter in range(1, 20):
        X = [[i, j] for i in range(20) for j in range(20)]

        kmeans = KMeans(n_clusters=4, max_iter=counter, n_init=1, random_state=10, init='random',
                        precompute_distances=False)
        pred = kmeans.fit_predict(X)

