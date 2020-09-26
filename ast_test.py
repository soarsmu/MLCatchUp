import torch
import numpy as np
from sklearn import __all__
import sklearn
from sklearn.cluster import KMeans

def someFunc(a, b, c):
    return a + b + c

print("HelloWorld")
sklearn_version = sklearn.__version__
np_version = np.__version__
concat_version = sklearn_version + np_version
print(concat_version)

hmm = sklearn.cluster
a = hmm.KMeans(n_jobs=3)
print(someFunc(1,2,someFunc(2,3,4)))
hehe = KMeans(n_clusters=2, random_state=0, n_jobs = 4, precompute_distances=True)
KMeans(n_jobs=3, n_clusters=3) if 3 > 2 else KMeans(n_clusters=1)
hehe.fit(x)

Pipeline(KMeans(n_jobs=3))
from sklearn.cluster import KMeans as km

km(n_jobs=3).fit([1,2,3])