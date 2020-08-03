import sklearn.cluster

arr = [[1,2], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1]]

a = sklearn.cluster.KMeans()
b = a.fit(arr)

from sklearn.cluster import KMeans as km
c = km()
d = km(8, n_jobs=4).fit(arr)
e = km().fit(X=arr).__str__()
f = sklearn.cluster.KMeans(sum(1,2)).fit(X=arr).__str__(1).__str__()

n_clust = 3
def n_job():
    return 2

class someClass:
    x=4
    def __init__(self):
        self.y=4

sklearn.cluster.KMeans(n_jobs=n_job(), n_clusters=n_clust)
sklearn.cluster.KMeans(n_jobs=someClass.someElse.x, n_clusters=someClass())
sklearn.cluster.KMeans(n_jobs="sss", n_clusters=5)

# Call
# Attribute
# Name
# Constant