import sklearn.cluster

arr = [[1,2], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1], [2,1]]

a = sklearn.cluster.KMeans()
b = a.fit(arr)

from sklearn.cluster import KMeans as km
c = km()
d = km(8, n_jobs=4).fit(arr)
e = km().fit(X=arr).__str__()
f = sklearn.cluster.KMeans(sum(1,2)).fit(X=arr).__str__(1).__str__()