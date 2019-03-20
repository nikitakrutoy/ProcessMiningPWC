import numpy as np
from scipy.spatial.distance import cdist


def in_cluster_distance(X, clusters, metric="euclidean", **kwargs):
    clusters_names = np.unique(clusters)
    distances = []
    for cluster_name in clusters_names:
        cluster = X[clusters == cluster_name]
        if "centroids" in kwargs.keys():
            centroid = kwargs["centroids"][cluster_name]
        else:
            centroid = np.mean(cluster, axis=0)
        distances.append(np.mean(cdist([centroid], cluster, metric=metric)))
    return distances


def between_cluster_distance(X, clusters, metric="euclidean", **kwargs):
    clusters_names = np.unique(clusters)
    distances = []
    for cluster_name in clusters_names:
        cluster = X[clusters == cluster_name]
        other_clusters = X[clusters != cluster_name]
        distances.append(np.mean(cdist(cluster, other_clusters, metric=metric)))
    return distances


def cluster_sizes(X, clusters, metric=None, **kwargs):
    sizes = []
    for cluster in np.unique(clusters):
        sizes.append(np.sum(clusters == cluster))
    return sizes


def matching_dissim(a, b):
    return np.sum(np.array(a) != np.array(b))
