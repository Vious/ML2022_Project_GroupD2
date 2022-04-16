from sklearn.cluster import KMeans
from kmeans import KMeansPP
from utils import cluster_acc

# data, label, num_class

kmeans = KMeansPP(data, num_class)
_, _, _, pred1 = kmeans.cluster()
acc1 = cluster_acc(label, pred1)

kmeans = KMeans(n_clusters=num_class, random_state=0).fit(data)
pred2 = kmeans.labels_
acc2 = cluster_acc(label, pred2)
print(acc1)
print(acc2)