# for the pca https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca



print(__doc__)



import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D





from sklearn.cluster import KMeans

from sklearn import datasets





from sklearn import preprocessing



from time import time

from sklearn import metrics

from sklearn.cluster import KMeans

from sklearn.datasets import load_digits

from sklearn.decomposition import PCA

from sklearn.preprocessing import scale

import sklearn.metrics

from sklearn.datasets import load_digits



def apply_Kmeans(full_dataset, number_clusters = 8):

    cltr = KMeans(n_clusters=number_clusters, init='random', n_init=1, max_iter=100, tol=0.001, precompute_distances=False,

                        verbose=0, random_state=None, copy_x=True, algorithm='full',  n_jobs=4)

    return vector_cluster



