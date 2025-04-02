# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 23:23:59 2020

@author: Susana Nascimento
"""
import numpy as np
# Pandas also for data management
import pandas as pd

## Anomalous Pattern Algorithm

def center_(x, cluster):
    """ 
    calculates the centroid of the cluster
    x - the original data matrix ( N x d)
    cluster - the set with indices (i= 1, 2, ..., N) of the objects belonging to the cluster
    returns the centroid of the cluster 
    """
    #number of columns
    mm = x.shape[1]
    centroidC = []
    
    for j in range(mm):
        zz = x[:, j]
        zc = []
        for i in cluster:
            zc.append(zz[i])
        centroidC.append(np.mean(zc))
    return centroidC


def distNorm(x ,remains, ranges, p):
    """ 
     Finds the normalized distances of data points in 'remains' to reference point 'p' 
     x - the original data matrix ( N x d)
     remains- the set of X-row indices: the indices of the entities under consideration
     ranges- the vector with the ranges of the data features  
     p - the reference data point the distances relate to
     distan- returns the column vetor  with the distances from p to remains 
     """

    
    mm = x.shape[1]      # number of data features
    rr = len(remains)    # number of entities in remains    
    z = x[remains, :]
    az = np.tile(np.array(p), (rr, 1))     # Construct an array by repeating input array np.array(p)  
                                           # the number of rows is rr
    
    rz = np.tile(np.array(ranges), (rr, 1))
    dz = (z - az) / rz
    dz = np.array(dz)
    ddz = dz * dz
    if mm > 1:
        di = sum(ddz.T)
    else:
        di = ddz.T
    distan = di
    return distan


def separCluster(x0, remains, ranges, a, b):
    """  
    Builds a cluster by splitting the points around the reference point 'a' from those around the reference point b 
    x0 - data matrix
    remains- the set of X-row indices: the indices of the entities under consideration
    ranges-  the vector with the ranges of the data features  
    a, b - the reference points
    cluster - returns a set with the row indices of the objects belonging to the cluster  
    """
    
    dista = distNorm(x0, remains, ranges, a)
    distb = distNorm(x0, remains, ranges, b)
    clus = np.where(dista < distb)[0]
    cluster = []
    for i in clus:
        cluster.append(remains[i])
    return cluster


 ## Consult description of building an Anomalous cluster (lecture K-means clustering)
def anomalousPattern(x, remains, ranges, centroid, me):
    """ Builds one anomalous cluster based on the algorithm 'Separate/Conquer' (B. Mirkin (1999): Machine Learning Journal) 
        x - data matrix,
        remains - the set of X-row indices: the indices of the entities under consideration
        ranges - normalizing values: the vector with ranges of data features  
        centroid - initial center of the anomalous cluster being build
        me - vector to shift the 0 (origin) to
        Returns a tuple with:
                cluster - set of (remains) row indices in the anomalous cluster, 
                centroid - center of the built anomalous cluster    
    """        
    key = 1
    while key == 1:
        cluster = separCluster(x, remains, ranges, centroid, me)
        if len(cluster) != 0:
            newcenter = center_(x, cluster)
          
        if  len([i for i, j in zip(centroid, newcenter) if i == j]) != len(centroid):
            centroid = newcenter
        else:
            key = 0
    return (cluster, centroid)

def dist(x, remains, ranges, p):
    """ 
      Calculates the normalized distances of data points in 'remains' to reference point 'p'   
       x - data matrix,
       remains - the set of X-row indices: the indices of the entities under consideration
       ranges - normalizing values: the vector with ranges of data features      
       distan - returns the calculated normalized distances
    """

    
    mm = x.shape[1]       #number of columns
    rr = len(remains)     # number of entities in remains  
    distan = np.zeros((rr,1))    
    for j in range(mm):
        z = x[:, j]         # j feature vector
        z = z.reshape((-1,1))
        zz = z[remains]
        y = zz - p[j]
        y = y / ranges[j]
        y = np.array(y)
        yy = y * y
        distan = distan + yy
    return distan



##### ****** Main body for the Iterative Anomalous Cluster Algorithm  *****
#### You should test and Validate it with the Market Towns Data set following the report
# Main execution adapted for Iris dataset
# ---------------------------
from sklearn.datasets import load_iris
# Load iris dataset
iris = load_iris()
x = iris.data.astype(np.float32)
# df = pd.DataFrame(x, columns=iris.feature_names)

nn = x.shape[0]  # number of data points
mm = x.shape[1]  # number of features

# Calculate global statistics
me = [np.mean(x[:, j]) for j in range(mm)]
mmax = [np.max(x[:, j]) for j in range(mm)]
mmin = [np.min(x[:, j]) for j in range(mm)]
ranges = []
normalization = 0  # set to 1 to ignore feature ranges, 0 otherwise
for j in range(mm):
    if normalization:
        ranges.append(1)
    else:
        rng = mmax[j] - mmin[j]
        if rng == 0:
            print("Variable num {} is constant!".format(j))
            rng = 1
        ranges.append(rng)

# Compute total data scatter on normalized data
sY = (x - me) / ranges
d = np.sum(sY ** 2)

# Iterative Anomalous Cluster Algorithm
remains = list(range(nn))  # indices of remaining points
threshold = 25  # minimum cluster size
numberC = 0  # counter of anomalous clusters

# Instead of a single array, store clusters as a list of dictionaries:
ancl = []  # each element: {'cluster': list, 'centroid': list, 'dD': float}

while len(remains) > 0:
    distance = dist(x, remains, ranges, me)
    # find the index of the point with the maximum distance from the overall mean
    ind = np.argmax(distance)
    index = remains[ind]
    centroid = x[index, :].tolist()  # initial anomalous center
    numberC += 1

    # Get anomalous cluster pattern
    cluster, centroid = anomalousPattern(x, remains, ranges, centroid, me)

    # If the cluster is empty, remove the farthest point and continue
    if len(cluster) == 0:
        remains.remove(index)
        continue

    # Standardize the centroid
    censtand = ((np.array(centroid) - np.array(me)) / np.array(ranges)).tolist()
    # Compute contribution dD; note: using len(cluster) to weight the cluster scatter.
    dD = np.sum((np.array(censtand) ** 2) * len(cluster) * 100 / d)

    # Remove cluster points from remains
    remains = list(set(remains) - set(cluster))

    # Store the cluster information
    ancl.append({
        'cluster': cluster,
        'centroid': censtand,
        'dD': dD
    })

# Filter clusters by threshold size
filtered_ancl = [ac for ac in ancl if len(ac['cluster']) >= threshold]

if len(filtered_ancl) == 0:
    print('Too great a threshold!!!')
else:
    # For demonstration, print out the clusters and their standardized centroids
    for i, ac in enumerate(filtered_ancl):
        print(f"Cluster {i + 1}:")
        print("  Size:", len(ac['cluster']))
        print("  Centroid (standardized):", np.round(ac['centroid'], 3))
        print("  Cluster contribution (%):", np.round(ac['dD'], 3))
