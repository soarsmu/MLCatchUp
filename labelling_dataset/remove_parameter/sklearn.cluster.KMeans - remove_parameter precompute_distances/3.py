#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 20:00:31 2019

@author: salauddinali
"""
from __future__ import print_function
import nltk
import pandas as pd
from nltk.corpus import gutenberg
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.cluster import k_means, hierarchical, AgglomerativeClustering,KMeans
from scipy.cluster.hierarchy import dendrogram, linkage  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#---------- 3D plot
import plotly
import plotly.graph_objs as go

def knnErrorEval(vector):
    num_clusters = 3
    num_seeds = 10
    max_iterations = 300
    labels_color_map = {
        0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073'
    }
    tsne_num_components = 2
    
    # texts_list = some array of strings for which TF-IDF is being computed
    
    # calculate tf-idf of texts
    #tf_idf_vectorizer = TfidfVectorizer(analyzer="word", use_idf=True, smooth_idf=True, ngram_range=(2, 3))
    #tf_idf_matrix = tf_idf_vectorizer.fit_transform(df_x)
    
    # create k-means model with custom config
    clustering_model = KMeans(
        n_clusters=num_clusters,
        max_iter=max_iterations,
        precompute_distances="auto",
        n_jobs=-1
    )
    