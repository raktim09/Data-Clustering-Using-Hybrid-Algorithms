

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
import math
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import datasets
import random #generate random nos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm
from sklearn import metrics
%matplotlib inline


###############################################################################

#loading iris datasets
iris=datasets.load_iris()
#print(iris)
iris.target

#X=#pd.DataFrame(iris.data,columns=['Sepal L','Sepal W','Petal L','Petal W'])
X=pd.read_csv("http://cs.joensuu.fi/sipu/datasets/MopsiLocations2012-Joensuu.txt",delimiter=r"\s+")
Y1=pd.DataFrame(iris.target,columns=['Target'])
#print(X.describe())
#print('SHAPE IS=', X.shape)
X.head()

#X2 saves the petal length and width array of all values as a matrix of size 150*2
X_data=X.iloc[:,[0,1]].values
X2=X.iloc[:,[0,1]].values

###############################################################################

start=time.time()


###############################################################################
m=X2.shape[0]   # No. of rows=150
n=X2.shape[1]   # No. of columns=2
m,n
n_iter=50

K=3   #No. of clusters

#Centroids is a n x K dimentional matrix, where each column will be a centroid for one cluster.
#Step 1:Initialize the centroids randomly from the data points:

Centroids=np.array([]).reshape(n,0)

for i in range(K):
    rand=random.randint(0,m-1)
    Centroids=np.c_[Centroids,X2[rand]]

#Step 2:a

Output={}

EuclidianDistance=np.array([]).reshape(m,0)
for k in range(K):
       tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
C=np.argmin(EuclidianDistance,axis=1)+1

#b

Y={}
for k in range(K):
    Y[k+1]=np.array([]).reshape(2,0)
for i in range(m):
    Y[C[i]]=np.c_[Y[C[i]],X2[i]]

for k in range(K):
    Y[k+1]=Y[k+1].T

for k in range(K):
     Centroids[:,k]=np.mean(Y[k+1],axis=0)

#Now we need to repeat step 2 till convergence is achieved.
#In other words, we loop over n_iter and repeat the step 2.a and 2.b as shown:
for i in range(n_iter):
     #step 2.a
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X2[i]]

      for k in range(K):
          Y[k+1]=Y[k+1].T

      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y

###############################################################################








##############################################################################


def MyDBSCAN(D,Centroid, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for i in Centroid:
      T=np.where(D == i)
      P=T[0][0]
      NeighborPts = regionQuery(D, P, eps)
      if len(NeighborPts) <= MinPts:
        labels[P] = -1   #changing -1 to 0
      else:
        C += 1
        growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    #for P in range(0, len(D)):
    #  if not (labels[P] == 0):
    #    continue

     # NeighborPts = regionQuery(D, P, eps)
     # if len(NeighborPts) < MinPts:
      #  labels[P] = -1
     # else:
     #   C += 1
     #   growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) > MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1

def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        if abs(math.sqrt((D[P][0]-D[Pn][0])*(D[P][0]-D[Pn][0])+(D[P][1]-D[Pn][1])*(D[P][1]-D[Pn][1]))) <= eps:
            neighbors.append(Pn)
    return neighbors

###############################################################################
Centroids=Centroids.tolist()
Centroid=[]
for i in range(K):
  temp=[]
  for j in range(2):
    temp.append(Centroids[j][i])
  temp[0]=float(str(temp[0])[:3])
  temp[1]=float(str(temp[1])[:3])
  Centroid.append(temp)
Centroid=np.array(Centroid)
# Run my HYBRID KMEANS DBSCAN implementation.
print('Running my implementation...')
start=time.time()
my_labels = MyDBSCAN(X_data,Centroid, eps=0.21, MinPts=4)
#print(my_labels)
plt.figure()
plt.scatter(X_data[:, 0], X_data[:, 1], c=my_labels)
plt.savefig('dbscan.png')
plt.title('DBSCAN PLOT')
print("Time taken by my implementation :- ",time.time()-start)

print("Silhoutte :- ")
print(metrics.silhouette_score(X_data,my_labels))



--------------------------------------------------
-------------------------------------------------
------------------------------
##KMEANS DBSCAN HYBRID WITH NEW DATASET
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
import math
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import datasets
import random #generate random nos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics as sm
from sklearn import metrics
%matplotlib inline


###############################################################################

#loading iris datasets
iris=datasets.load_iris()
#print(iris)
iris.target

#X=#pd.DataFrame(iris.data,columns=['Sepal L','Sepal W','Petal L','Petal W'])
X=pd.read_csv("http://cs.joensuu.fi/sipu/datasets/MopsiLocations2012-Joensuu.txt",delimiter=r"\s+")
Y1=pd.DataFrame(iris.target,columns=['Target'])
#print(X.describe())
#print('SHAPE IS=', X.shape)
X.head()

#X2 saves the petal length and width array of all values as a matrix of size 150*2
X_data=X.iloc[:,[0,1]].values
X2=X.iloc[:,[0,1]].values

###############################################################################

start=time.time()


###############################################################################
m=X2.shape[0]   # No. of rows=150
n=X2.shape[1]   # No. of columns=2
m,n
n_iter=50

K=3   #No. of clusters

#Centroids is a n x K dimentional matrix, where each column will be a centroid for one cluster.
#Step 1:Initialize the centroids randomly from the data points:

Centroids=np.array([]).reshape(n,0)

for i in range(K):
    rand=random.randint(0,m-1)
    Centroids=np.c_[Centroids,X2[rand]]

#Step 2:a

Output={}

EuclidianDistance=np.array([]).reshape(m,0)
for k in range(K):
       tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
       EuclidianDistance=np.c_[EuclidianDistance,tempDist]
C=np.argmin(EuclidianDistance,axis=1)+1

#b

Y={}
for k in range(K):
    Y[k+1]=np.array([]).reshape(2,0)
for i in range(m):
    Y[C[i]]=np.c_[Y[C[i]],X2[i]]

for k in range(K):
    Y[k+1]=Y[k+1].T

for k in range(K):
     Centroids[:,k]=np.mean(Y[k+1],axis=0)

#Now we need to repeat step 2 till convergence is achieved.
#In other words, we loop over n_iter and repeat the step 2.a and 2.b as shown:
for i in range(n_iter):
     #step 2.a
      EuclidianDistance=np.array([]).reshape(m,0)
      for k in range(K):
          tempDist=np.sum((X2-Centroids[:,k])**2,axis=1)
          EuclidianDistance=np.c_[EuclidianDistance,tempDist]
      C=np.argmin(EuclidianDistance,axis=1)+1
     #step 2.b
      Y={}
      for k in range(K):
          Y[k+1]=np.array([]).reshape(2,0)
      for i in range(m):
          Y[C[i]]=np.c_[Y[C[i]],X2[i]]

      for k in range(K):
          Y[k+1]=Y[k+1].T

      for k in range(K):
          Centroids[:,k]=np.mean(Y[k+1],axis=0)
      Output=Y

###############################################################################








##############################################################################


def MyDBSCAN(D,Centroid, eps, MinPts):
    labels = [0]*len(D)
    C = 0
    for i in Centroid:
      T=np.where(D == i)
      P=T[0][0]
      NeighborPts = regionQuery(D, P, eps)
      if len(NeighborPts) <= MinPts:
        labels[P] = -1   #changing -1 to 0
      else:
        C += 1
        growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    #for P in range(0, len(D)):
    #  if not (labels[P] == 0):
    #    continue

     # NeighborPts = regionQuery(D, P, eps)
     # if len(NeighborPts) < MinPts:
      #  labels[P] = -1
     # else:
     #   C += 1
     #   growCluster(D, labels, P, NeighborPts, C, eps, MinPts)
    return labels

def growCluster(D, labels, P, NeighborPts, C, eps, MinPts):
    """
    Parameters:
      `D`      - The dataset (a list of vectors)
      `labels` - List storing the cluster labels for all dataset points
      `P`      - Index of the seed point for this new cluster
      `NeighborPts` - All of the neighbors of `P`
      `C`      - The label for this new cluster.
      `eps`    - Threshold distance
      `MinPts` - Minimum required number of neighbors
    """
    labels[P] = C
    i = 0
    while i < len(NeighborPts):
        Pn = NeighborPts[i]
        if labels[Pn] == -1:
           labels[Pn] = C
        elif labels[Pn] == 0:
            labels[Pn] = C
            PnNeighborPts = regionQuery(D, Pn, eps)
            if len(PnNeighborPts) > MinPts:
                NeighborPts = NeighborPts + PnNeighborPts
        i += 1

def regionQuery(D, P, eps):
    neighbors = []
    for Pn in range(0, len(D)):
        if abs(math.sqrt((D[P][0]-D[Pn][0])*(D[P][0]-D[Pn][0])+(D[P][1]-D[Pn][1])*(D[P][1]-D[Pn][1]))) <= eps:
            neighbors.append(Pn)
    return neighbors

###############################################################################
Centroids=Centroids.tolist()
Centroid=[]
for i in range(K):
  temp=[]
  for j in range(2):
    temp.append(Centroids[j][i])
  temp[0]=float(str(temp[0])[:3])
  temp[1]=float(str(temp[1])[:3])
  Centroid.append(temp)
Centroid=np.array(Centroid)
# Run my HYBRID KMEANS DBSCAN implementation.
print('Running my implementation...')
start=time.time()
my_labels = MyDBSCAN(X_data,Centroid, eps=0.25, MinPts=4)
#print(my_labels)
plt.figure()
plt.scatter(X_data[:, 0], X_data[:, 1], c=my_labels)
plt.savefig('dbscan.png')
plt.title('DBSCAN PLOT')
print("Time taken by my HYBRID KMEANS DBSCAN implementation :- ",time.time()-start)

print("Silhoutte :- ")
print(metrics.silhouette_score(X_data,my_labels))
