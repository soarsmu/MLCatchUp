"""
.. module:: unsupervised
   :synopsis: This file is meant to reproduce the unsupervised results, thus it is not extensible and only works
              for the particular datasets we used. The methods, however, are very flexible and can be used on a variety
              of datasets.
"""
import os
import time
import numpy as np
import pandas as pd
from cssnormaliser import CSSNormaliser
from sklearn.manifold import Isomap,locally_linear_embedding,spectral_embedding,MDS,TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist,pdist
#import skbio
import sklearn.cluster as cluster
import hdbscan
import sklearn.preprocessing
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from pyclustering.cluster import kmedians,kmedoids

# Finding the directory of the project folder
dirname = os.path.dirname(__file__)
project_dir = os.path.join(dirname,"..","..")
path = os.path.join(project_dir,"data","processed")
figure_path = os.path.join(project_dir,"reports","figures")
wwfdf = pd.read_csv(filepath_or_buffer=path+"/wwfdf",encoding="ISO-8859-1",index_col = 0)
riverdf = pd.read_csv(path+"/riverdf",index_col=0)
riverdfCss = CSSNormaliser(log=False).fit_transform(riverdf)
riverdfCssLog = CSSNormaliser(log=True).fit_transform(riverdf)
fulldf = pd.read_csv(path+ "/fulldf",index_col=0)
fulldfCssLog = CSSNormaliser(log=True).fit_transform(fulldf)

day_time_string = time.strftime("%Y-%m-%d")

def edge_colors():
    edgecolors = np.array([[1,0,0]]*164,dtype = float)
    edgecolors[wwfdf.Trip ==2] = [0,1,0]
    edgecolors[wwfdf.Trip ==3] =[0,0,1]
    return edgecolors

def add_jitter(arr):
    np.random.seed(1125)
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def cluster_on_map(labels, title="Clustering", **kwargs):
    plt.subplots(figsize=(7, 7))
    X, Y = (add_jitter(wwfdf.Easting), add_jitter(wwfdf.Northing))
    edge_array = (edge_colors())
    if labels.min() == -1:
        palette = [(0, 0, 0)] + sns.color_palette("Set2", np.unique(labels).shape[0] - 1)


    else:
        palette = sns.color_palette("Set2", np.unique(labels).shape[0])
    sns.scatterplot(x=X, y=Y, hue=labels, palette=palette,
                    style=wwfdf.Water, s=100 + 20 * wwfdf.Trip, edgecolor=(0, 0, 0), linewidth=0.5,
                    alpha=0.8)
    plt.title(title)
    if kwargs.get("fname"):
        plt.savefig(fname=figure_path + kwargs["fname"], dpi=300)

def list_of_clusters_to_clusters(list_of_clusters,y = np.zeros(164,dtype=int)):
    """
    creates ndarray from list of groups of clusters
    eg list of clusters looks like this:
    [[1,5,6],[2,4,3].[7,8]] meaning that samples 1,5,6 belong to the same group
    samples 2,3,4 belong to the same group and samples 7,8 to the same.
    the output would be
    np.array([0,1,1,1,0,0,2,2])
    :param list_of_clusters:
    :param y:
    :return:
    """
    for j,i in enumerate(list_of_clusters):
        y[i]=int(j)
    return y


def KMedians(partitions,dataframe, iterations =20):
    """
    performs KMedians clustering with "partitions" clusters and  "iterations"
    :param iterations:
    :param partitions:
    :param dataframe:
    :return:
    """
    np.random.seed(11235)
    km_instance = kmedians.kmedians(((dataframe)*1).to_numpy(),dataframe.sample(n=partitions).to_numpy(),itermax = 100)
    km_instance.process()
    error =(km_instance.get_total_wce())
    print(error)

    for j in range(iterations):
        kmedoids_instance = kmedians.kmedians(((dataframe)*1).to_numpy(),dataframe.sample(n=partitions).to_numpy(),itermax = 100)
    #     kmedoids_instance = kmedoids.kmedoids(((riverdfCssLog)*1).to_numpy(),np.random.randint(0,165,size=2),itermax = 10000)
        kmedoids_instance.process()
        new_error =(kmedoids_instance.get_total_wce())
        if new_error < error:
            km_instance = kmedoids_instance
            error = new_error
    return list_of_clusters_to_clusters(km_instance.get_clusters())

# if the results directory does not exist then create it
os.makedirs(os.path.join(project_dir,"results","unsupervised"), exist_ok=True)

resultdf = pd.DataFrame(index=wwfdf.index)
#resultdf = pd.read_csv("../../results/unsupervised/unsupervised.csv",header = 0,index_col=0)
resultdf["KMeans7rivcsslogDetails"] = "KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdfCssLog))"
resultdf["KMeans7rivcsslog"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdfCssLog)).labels_

resultdf["KMeans2rivcsslogDetails"] =  "KMeans(random_state=11235,n_jobs = -1,n_clusters=2).fit((riverdfCssLog)).labels_"
resultdf["KMeans2rivcsslog"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=2).fit((riverdfCssLog)).labels_

resultdf["KMeans2fullcsslogDetails"] =  "KMeans(random_state=11235,n_jobs = -1,n_clusters=2).fit((fulldfCssLog)).labels_"
resultdf["KMeans2fullcsslog"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=2).fit((fulldfCssLog)).labels_

resultdf["KMeans3fullcsslogDetails"] = "KMeans(random_state=11235,n_jobs = -1,n_clusters=3).fit((fulldfCssLog))"
resultdf["KMeans3fullcsslog"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=3).fit((fulldfCssLog)).labels_

resultdf["KMeans7rivDetails"] =  "KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdf)).labels_"
resultdf["KMeans7riv"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdf)).labels_

resultdf["KMeans7rivcssDetails"] =  "KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdfCss)).labels_"
resultdf["KMeans7rivcss"] =  cluster.KMeans(random_state=11235,n_jobs = -1,n_clusters=7).fit((riverdfCss)).labels_

resultdf["HDBSCAN4rivcsslogDetails"] = "hdbscan.HDBSCAN(min_cluster_size=4,metric=\"euclidean\").fit_predict(riverdfCssLog)"
resultdf["HDBSCAN4rivcsslog"] = hdbscan.HDBSCAN(min_cluster_size=4,metric="euclidean").fit_predict(riverdfCssLog)

resultdf["HDBSCAN4fullcsslogDetails"] = "hdbscan.HDBSCAN(min_cluster_size=4,metric=\"euclidean\").fit_predict(fulldfCssLog)"
resultdf["HDBSCAN4fullcsslog"] = hdbscan.HDBSCAN(min_cluster_size=4,metric="euclidean").fit_predict(fulldfCssLog)
#
# resultdf["KMedians2rivcsslogDetails"] = "KMedians(partitions=2,dataframe =riverdfCssLog)"
# resultdf["KMedians2rivcsslog"] = KMedians(partitions=2,dataframe =riverdfCssLog)
#
# resultdf["KMedians2fullcsslogDetails"] = "KMedians(partitions=2,dataframe=fulldfCssLog)"
# resultdf["KMedians2fullcsslog"] = KMedians(partitions=2,dataframe=fulldfCssLog)
#
# resultdf["KMedians7fullcsslogDetails"] = "KMedians(partitions = 7,dataframe =fulldfCssLog)"
# resultdf["KMedians7fullcsslog"] = KMedians(partitions = 7,dataframe =fulldfCssLog)
#
# cluster_on_map(resultdf["KMeans7rivcsslog"],title="KMeans 7 river set")

#
#
# resultdf["KMedians7rivcsslogDetails"] = "KMedians(partitions = 7,dataframe =riverdfCssLog)"
# resultdf["KMedians7rivcsslog"] = KMedians(partitions = 7,dataframe =riverdfCssLog)

# save result as a csv in peru-rivers/results/unsupervised/unsupervised.csv
resultdf.to_csv(os.path.join(project_dir,"results","unsupervised","unsupervised"+day_time_string+".csv"))