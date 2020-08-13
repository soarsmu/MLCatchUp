from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import csv

import getpass
import os
from pathlib import Path
import shutil as s
import time
from multiprocessing import cpu_count
'''
     The silhouette_score gives the average value for all the samples.
     This gives a perspective into the density and separation of the formed clusters
'''
def Silhoette_Coeff(pixel_values_of_all_images,filenames):
    
    print("Computting Silhoette Coefficient ...")
    print("")
    
    silhouette = []
    for i in range(len(pixel_values_of_all_images)):
        pixel_value = pixel_values_of_all_images[i]
     
        clusterer = KMeans(n_clusters = 3, n_jobs = 1)
        cluster_labels = clusterer.fit_predict(pixel_value)
        silhouette_avg = silhouette_score(pixel_value, cluster_labels)
        print("The average silhouette_score is :", silhouette_avg)         
        
        silhouette.append(silhouette_avg)
        
        
    path = os.getcwd()
    parent_path = Path(path).parent
    os.chdir(parent_path)
    for files in os.listdir():  
        if files == 'Metric.csv': 
            os.remove('Metric.csv')    
        
    else: 
        file = 'Metrics'
        with open(file + '.csv' , 'a' ,newline='') as csvfile :
            writer = csv.writer(csvfile)
            writer.writerow(['FILENAME','SILHOETTE COEFFICIENT']) 
            
            for i in range(0,len(filenames)):
                filename = filenames[i]
                metric = silhouette[i]
                
                writer.writerow([filename,metric])
    os.chdir(path)

    return 1


def Davis_Bouldin(pixel_values_of_all_images,filenames): 
    
    print("Computting Davies Bouldin Index ...")
    print("")
    
    DB = []
    for i in range(len(pixel_values_of_all_images)):
        pixel_value = pixel_values_of_all_images[i]    
        clusterer = KMeans(n_clusters = 3, n_jobs = -1)
        cluster_labels = clusterer.fit_predict(pixel_value)
        Davis_B = davies_bouldin_score(pixel_value, cluster_labels)
        print("File ...",filenames[i],"      The davis Bouldin score is :", Davis_B)         
        DB.append(Davis_B) 
        
    path = os.getcwd()
    parent_path = Path(path).parent
    os.chdir(parent_path)
    for files in os.listdir():  
        if files == 'Davis Bouldin Metrics': 
            os.remove('Metric.csv')    
        
    else: 
        file = 'Davis Bouldin Metrics'
        with open(file + '.csv' , 'a' ,newline='') as csvfile :
            writer = csv.writer(csvfile)
            writer.writerow(['FILENAME','SILHOETTE COEFFICIENT']) 
            
            for i in range(0,len(filenames)):
                filename = filenames[i]
                metric = DB[i]
                
                writer.writerow([filename,metric])
    os.chdir(path)

    return 1
