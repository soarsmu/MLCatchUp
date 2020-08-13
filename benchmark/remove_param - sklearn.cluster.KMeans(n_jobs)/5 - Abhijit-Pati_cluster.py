import numpy as np
import pickle
from scipy.spatial import distance_matrix
import codecs
import csv
import sklearn
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from numpy import save

def calculate_WSS(points, kmin, kmax):
  sse = []  
  sil = [] 
  clusters=[]
  for k in tqdm(range(kmin, kmax+1)):
    c=[]
    kmeans = KMeans(n_clusters = k, init='k-means++',algorithm='full',max_iter=2000,n_init=50).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    labels = kmeans.labels_
    for cent in centroids:
      d=[]
      for p in points:
        d.append(np.dot(cent,p))
      dmax=max(d)
      dmaxi=d.index(dmax)
      c.append([dmaxi,labels[dmaxi]])
    curr_sse = 0    
    # calculating the square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      diff=points[i]-curr_center
      curr_sse += np.linalg.norm(diff)**2
      
    sse.append(curr_sse)
    sil.append(silhouette_score(points, labels, metric = 'euclidean'))
    clusters.append([labels,c])
  return sse,sil,clusters

# Xt = [ x.strip().split() for x in codecs.open("data/title_vectors.txt", "r", "utf-8").readlines() ]
# Xt = np.array(Xt).astype(np.float64)
# Xc = [ x.strip().split() for x in codecs.open("data/vectors.txt", "r", "utf-8").readlines() ]
# Xc = np.array(Xc).astype(np.float64)
# X=[]
# for i in range(len(Xt)):
#     X.append([])
#     for j in range(len(Xt[i])):
#         X[i].append(Xt[i][j])
#     for k in range(len(Xc[i])):
#         X[i].append(Xc[i][k])

X = [ x.strip().split() for x in codecs.open("data/ap_c_vectors.txt", "r", "utf-8").readlines() ]
X = np.array(X).astype(np.float64)
print(len(X))
# eigen_solver='amg'
# clustering = sklearn.cluster.SpectralClustering(n_clusters=12,assign_labels="discretize",random_state=0,n_jobs=-1).fit(X)
# labels = clustering.labels_
# # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_clusters_ = len(set(labels))
# n_noise_ = list(labels).count(-1)


# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
nX=[list(x/np.linalg.norm(x)) for x in X]
print(len(nX))
Dmat=distance_matrix(nX, nX)
print("|WSS and Silhouette score plotter|")
kmin=int(input("Enter value of kmin: "))
kmax=int(input("Enter value of kmax: "))
sse,sil,clusters=calculate_WSS(nX,kmin,kmax)
k=range(kmin, kmax+1)
silp=[k,sil]
ssep=[k,sse]
with open("sil.csv","w") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(silp)
with open("sse.csv","w") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(ssep)
kmin2=kmin
kmax2=kmax
ans='y'
while ans=='y':
  kmini=k.index(kmin2)
  kmaxi=k.index(kmax2)+1
  plt.plot(k[kmini:kmaxi],sse[kmini:kmaxi],label='sse')
  plt.scatter(k[kmini:kmaxi],sse[kmini:kmaxi],label='sse')
  plt.savefig('WSS.png')
  plt.clf()
  plt.plot(k[kmini:kmaxi],sil[kmini:kmaxi],label='sil')
  plt.scatter(k[kmini:kmaxi],sil[kmini:kmaxi],label='sse')
  plt.savefig('Sil.png')
  plt.clf()
  ans=input("Do you want to continue(y/n)? : ")
  if ans=='y':
    kmin2=int(input("Enter value of kmin: "))
    kmax2=int(input("Enter value of kmax: "))

kopt=int(input("Enter optimum value of k based on WSS and Silhouette score plot: "))
kindex=k.index(kopt)
labels=clusters[kindex][0]
centroids=clusters[kindex][1]
with open("clusters.txt", "wb") as fp:
  pickle.dump(clusters[kindex], fp)
with open("clusters.txt", "rb") as fp:
  ldata=pickle.load(fp)
print(ldata)
save('Dmat.npy',Dmat)
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)


# db = DBSCAN(eps=10, min_samples=5 ,n_jobs=-1).fit(X)
# labels = db.labels_
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)


print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
A=[]
for i in range(n_clusters_):
    A.append([])
for i in range(len(X)):
    if labels[i]!=-1:
        A[labels[i]-1].append(i)
for i in range(n_clusters_ ):
    print("No of Article in part ",i," :",len(A[i]))
