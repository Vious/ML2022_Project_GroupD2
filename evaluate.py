import argparse
import matplotlib.pyplot as plt
from ast import parse
from sklearn import manifold
from sklearn.cluster import KMeans
from kmeans import KMeansPP
from utils import cluster_acc
from sklearn import metrics
from sklearn.datasets import make_blobs


def main(args):
    num_class = args.classes
    num_samples = args.samples
    # X, Y = make_blobs(n_samples=num_samples,n_features=2,centers=[[1, 1],[1,2],[2,1],[2,2], \
    #     [-1,-1],[-1,-2],[-2,-1],[-2,-2], [-1,1],[-1,2],[-2,1],[-2,2], [1, -1], [1, -2], [2, -1], [2, -2]], 
    #              cluster_std=[0.05,0.1,0.1,0.05, 0.3,0.2,0.2,0.1, 0.2,0.3,0.1,0.2, 0.1,0.3,0.2,0.3],random_state=args.seed)
    X, Y = make_blobs(n_samples=num_samples,n_features=2,centers=[[1, 1],[2,2],[3,3],[4,4]], \
            cluster_std=[0.1,0.2,0.1,0.05],random_state=args.seed)
    
    # data, label, num_class
    label = Y
    data = X
    plt.scatter(X[:,0],X[:,1],marker='o',alpha=0.5)
    plt.show()
    kmeans = KMeansPP(data, num_class, args.seed)
    _, _, _, pred1 = kmeans.cluster()
    acc1 = cluster_acc(label, pred1)

    # plt.subplot(4,4,index+1)           
    plt.scatter(X[:,0],X[:,1],c=pred1[:],alpha=0.5, s=5)
    score = metrics.calinski_harabasz_score(X,pred1)
    print('calinski harabasz score:', score)
    plt.text(.99,.01,('clusters=%d score:%.2f')%(num_class,score),  
            transform=plt.gca().transAxes,size=10,   
            horizontalalignment='right')
    plt.show()


    kmeans = KMeans(n_clusters=num_class, random_state=args.seed).fit(data)
    pred2 = kmeans.labels_
    plt.scatter(X[:,0],X[:,1],c=pred2[:],alpha=0.5, s=5)
    score = metrics.calinski_harabasz_score(X,pred2)
    print('calinski harabasz score:', score)
    plt.text(.99,.01,('clusters=%d score:%.2f')%(num_class,score),  
            transform=plt.gca().transAxes,size=10,   
            horizontalalignment='right')
    plt.show()
    acc2 = cluster_acc(label, pred2)
    print('Mean accuracy of our kmeans implementation:', acc1)
    print('Mean accuracy of sklearn kmeans:', acc2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', type=int, default=8, help='number of classes defined for clustering')
    parser.add_argument('--samples', type=int, default=2000, help='number of samples to generate')
    parser.add_argument('--seed', type=int, default=2022, help='random seed for generating data')
    parser.add_argument('--opt', type=str, default='axis', help='')
    args = parser.parse_args()
    main(args)
