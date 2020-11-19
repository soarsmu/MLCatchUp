# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as km
from sklearn.cluster import MiniBatchKMeans as minikm
from sklearn.cluster import Birch
import pandas as pd
from sklearn import metrics
import time
from sklearn.decomposition import PCA
from pyclustering.cluster import kmeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import davies_bouldin_score

def best_original_kmeans(number_clusters, data):

    for i in range(1, number_clusters + 1):

        '''Utiliza K-means para clusterizar os dados'''
        kmeans = km(n_clusters=i, n_jobs=-1, n_init=20, verbose=False, tol=0.000001)
        kmeans.fit(X=data)

    return clusters, cost_clusters, best_model, best_cluster