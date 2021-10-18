'''
Name - Anmol Bishnoi
Roll Number - B19069
Branch - CSE
Mobile Number - 7042845211
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

test = pd.read_csv("mnist-tsne-test.csv")
train = pd.read_csv("mnist-tsne-train.csv")

def purity_score(y, pred):                                      #func for purity score from snippets
    contingency_matrix=metrics.cluster.contingency_matrix(y, pred)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

def dbscan_predict(dbscan_model, X_new):                        #func for prediction after dbscan
    y_new = np.ones(shape=len(X_new), dtype=int)*-1
    metric = spatial.distance.euclidean
    for j, x_new in X_new.iterrows():
        for i, x_core in enumerate(dbscan_model.components_):
            if metric(x_new, x_core) < dbscan_model.eps:
                y_new[j] = dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                break
    return y_new


#Q1

def kmean(test, k):                                             #func for K-Means Clustering
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(train[train.columns[:2]])
    kmeans_prediction = kmeans.predict(test[test.columns[:2]])

    plt.scatter(test[test.columns[0]], test[test.columns[1]], c = kmeans_prediction, cmap = 'rainbow', s = 13)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], c='black')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

    print('Purity Score =', purity_score(test[test.columns[-1]], kmeans_prediction))

#part A and B
print('K-means Clustering on Training Data')
kmean(train, 10)

#part C and D
print('\nK-means Clustering on Testing Data')
kmean(test, 10)


#Q2

def gm(test, n):                                                #func for GMM based Clustering
    gmm = GaussianMixture(n_components = n, random_state = 42)
    gmm.fit(train[train.columns[:2]])
    gmm_prediction = gmm.predict(test[test.columns[:2]])

    plt.scatter(test[test.columns[0]], test[test.columns[1]], c = gmm_prediction, cmap = 'rainbow', s = 13)
    plt.scatter(gmm.means_[:,0], gmm.means_[:,1], c='black')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

    print('Purity Score =', purity_score(test[test.columns[-1]], gmm_prediction))

#part A and B
print('\nGMM based Clustering on Training Data')
gm(train, 10)

#part C and D
print('\nGMM based Clustering on Testing Data')
gm(test, 10)


#Q3

def dbs(X, e, m):                                               #func for DBSCAN
    dbscan_model = DBSCAN(eps = e, min_samples = m).fit(train[train.columns[:2]])
    
    y_new = dbscan_predict(dbscan_model, X[X.columns[:2]])

    drop_list=[]
    for i in range(len(y_new)):
        if y_new[i] == -1:
            drop_list.append(i)
    
    X_new = X.copy()
    X_new.drop(drop_list, inplace = True) 
    y_new = np.delete(y_new, drop_list)
    
    plt.scatter(X_new[X_new.columns[0]], X_new[X_new.columns[1]], c = y_new, cmap = 'rainbow', s = 13)
    plt.title('DBSCAN for eps = ' + str(e) + ' and min_samples = ' + str(m))
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

    print('\nEpsilon =', e, 'MinPoints =', m)
    print('Purity Score =', purity_score(X_new[X_new.columns[-1]], y_new))
    print('No. of clusters formed =', len(set(y_new)))

#part A and B
print("\nDBSCAN on Training Data")
dbs(train, 5, 10)

#part C and D
print("\nDBSCAN on Testing Data")
dbs(test, 5, 10)


#Bonus Question A

K=[2, 5, 8, 12, 18, 20]

distortion=[]
for k in K:
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(train[train.columns[:2]])
    distortion.append(kmeans.inertia_)

plt.plot(K, distortion)
plt.title("Elbow Method on K-Means Clustering")
plt.xticks(K)
plt.xlabel("Values of K")
plt.ylabel("Distortion")
plt.show()

distortion=[]
for k in K:
    gmm = GaussianMixture(n_components = k, random_state = 42)
    gmm.fit(train[train.columns[:2]])
    distortion.append(gmm.score(train[train.columns[:2]])*len(train))

plt.plot(K, distortion)
plt.title("Elbow Method on GMM based Clustering")
plt.xticks(K)
plt.xlabel("Values of K")
plt.ylabel("Total Log Likelihood")
plt.show()


#Bonus Question B

for e in [1, 5, 10]:
    for m in [1, 10, 30, 50]:
        dbs(train, e, m)