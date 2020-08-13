import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np

from pca import *
from util import *


def difference(lst):
    return [lst[i + 1] - lst[i] for i in range(len(lst) - 1)]
    
def best_accelation(Js):
    J1 = difference(Js)
    J2 = difference(J1)
    best = np.argmin(J2) + 2
    return best
    
def test_kmeans():
    X = loadmat('ex7/ex7data2.mat')['X']
    n_init = int(X.shape[0] ** 0.5)
    Js = []
    Ks = range(1, 11)
    for k in Ks:
        km = KMeans(n_clusters=k, n_init=n_init, n_jobs=-1)
        km.fit(X)
        Js.append(km.score(X))
#    plotJ(Ks,Js)
    bestK = best_accelation(Js)
    km = KMeans(n_clusters=bestK, n_init=n_init, n_jobs=-1)
    km.fit(X)
    plt.clf()
    plt.scatter(*np.split(X, 2, axis=1), c='g')
    plt.scatter(*np.split(km.cluster_centers_, 2, axis=1), c='b', marker='D')
    plt.show()
    
def plotJ(Ks, Js):
    plt.xlim(xmin=1)
    plt.plot(Ks, Js, '*-')
    plt.xlabel('K')
    plt.ylabel('cost')
    plt.show()  

    
def image_compress(wait=False):
    img = loadmat('ex7/bird_small.mat')['A']
    X = img.reshape(-1, 3)
    km = KMeans(n_clusters=16, n_jobs=-1)
    km.fit(X)

    normed = np.array([km.cluster_centers_[i] for i in km.predict(X)], dtype=img.dtype)
    nimg = normed.reshape(img.shape)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.imshow(img)
    ax2.imshow(nimg)
    fig.show()

def test_pca():
    X = loadmat('ex7/ex7data1.mat')['X']
    # precison is affected by z_scaling, do normalization first
    X = z_scale(X)[0]
    k, Z, Xt = libPCA(X)
#    k, Z, Xt = myPCA(X) 
    
    plt.scatter(X[:, 0], X[:, 1])
    plt.scatter(Xt[:, 0], Xt[:, 1], color='red')
    plt.show()

def show_face(m, ax):
    ax.imshow(m.reshape((32, 32)).T, plt.cm.gray)

def faces():
    X = loadmat('ex7/ex7faces.mat')['X']
    pca = PCA(2)
    Y = pca.fit_transform(scale(X))
    Xp = pca.inverse_transform(Y)
    fig = plt.figure()
    for i in range(4):
        ax1 = fig.add_subplot(i+1, 2, 1)
        ax2 = fig.add_subplot(i+1, 2, 2)
        show_face(X[i],ax1)
        show_face(Xp[i],ax2)
    fig.show()
    raw_input('w')


if __name__ == '__main__':
#     test_pca()
    faces()
