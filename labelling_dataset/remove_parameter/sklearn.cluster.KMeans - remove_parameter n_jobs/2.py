# http://www.awesomestats.in/python-cluster-validation/



import umap

import time

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import fetch_mldata

import plotly.plotly as py

from sklearn.cluster import KMeans

from sklearn.cluster import MiniBatchKMeans

import sklearn.cluster as cluster

from sklearn.metrics import silhouette_score

import numpy as np

from scipy.spatial.distance import cdist,pdist

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from scipy.cluster.vq import kmeans

from matplotlib import cm

from sklearn import metrics

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler

import pandas_profiling

# Silhouette score

def silhouette_score(x):

    for n_cluster in range(2, 21):

        kmeans = KMeans(n_clusters=n_cluster, n_jobs=-1).fit(x)

        label = kmeans.labels_

        sil_coeff = metrics.silhouette_score(x, label, metric='euclidean')

        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
 



kmeans_model = KMeans(n_clusters=8, n_jobs=-1).fit(random_sample)

y_kmeans = kmeans_model.predict(random_sample)
