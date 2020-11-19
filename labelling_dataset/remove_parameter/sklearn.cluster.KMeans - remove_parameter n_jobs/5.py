from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster
import numpy
import time


class ClusteringAlgorithmPool:


    # Setting up variables and objects.
    def __init__(self, n_cluster, random_state, n_jobs, dist_metric, linkage_method, eps_dbscan, min_pts_dbscan):
        self.n_cluster = n_cluster
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.dist_metric = dist_metric
        self.runtime = numpy.zeros(4)

        # K means
        self.kmeans_obj = KMeans(n_clusters=self.n_cluster, max_iter=100, random_state=self.random_state,
                                 n_jobs=self.n_jobs)
       
   