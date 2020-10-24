import random

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial import distance
from collections import defaultdict
from sklearn.cluster import KMeans
import sys
from main.utils import print_dict

centroid_assignments = set()
count = 0


def normalize_between_0_and_1(x):
    x_max, x_min = x.max(), x.min()
    # print("x_max: ", x_max, ", x_min: ", x_min)
    x = (x - x_min) / (x_max - x_min) if x_max != x_min else x
    return x


def get_euclidean_distance(v1, v2):
    d = distance.euclidean(v1, v2)
    return d


# initialize centroids with any random values / points
def randomly_initialize_cluster_centroids(k, matrix):
    centroids = {}
    for i in range(k):
        centroids[i] = np.random.rand(len(matrix))
    return centroids


# assign random points in the sim_matrix as centroids
def initialize_cluster_centroids(k, matrix):
    k_list = random.sample(range(len(matrix)), k)
    global count
    while set(k_list) in centroid_assignments:
        count = count + 1
        k_list = random.sample(range(len(matrix)), k)
    centroid_assignments.add(frozenset(k_list))
    centroids = {}
    for k_idx, k_value in enumerate(k_list):
        centroids[k_idx] = matrix[k_value]
    return centroids


# given the cluster and the points in that cluster, compute the cluster centroids
def compute_cluster_centroids(clusters: dict, matrix) -> dict:
    centroids = {}
    for cluster_idx, cluster_points in clusters.items():
        arr = []
        for point in cluster_points:
            arr.append(matrix[point])
        arr = np.array(arr)
        centroids[cluster_idx] = np.mean(arr, axis=0)
    return centroids


# check whether two cluster assignments are the same
def are_clusters_same(c1, c2):
    # print("Old: ", c1)
    # print("New: ", c2)
    return c1 == c2


# get average distance of all points in a cluster from its centroid
def get_distance_from_centroid(centroid, point_index_list, matrix):
    dist_sum, num_points = 0, len(point_index_list)
    for idx in point_index_list:
        dist_sum += get_euclidean_distance(centroid, matrix[idx])
    return dist_sum / num_points if num_points != 0 else 0  ### check?


# get average distance of all points from their centroids in a given cluster assignment
def get_average_clustering_distance(clusters: dict, matrix):
    k_centroids = compute_cluster_centroids(clusters, matrix)
    dist_sum, k = 0, len(k_centroids.keys())
    for i, val in k_centroids.items():
        dist_sum += get_distance_from_centroid(val, clusters[i], matrix)
    return dist_sum / k


# run the K means algorithm for multiple random starts and return the clusters with least avg distance
def multiple_random_starts_k_means(matrix, k: int, max_iters: int, random_starts: int) -> dict:
    # initialize the min_clusters and min_avg_distance between the centroids and the points
    min_clusters, min_avg_dist = {}, float("inf")

    while random_starts > 0:
        clusters = k_means(matrix, k, max_iters)

        # compute the average distance of all points from their cluster centroids
        avg_cluster_distance = get_average_clustering_distance(clusters, matrix)

        # update the min clusters and min avg distances
        min_clusters = clusters if avg_cluster_distance < min_avg_dist else min_clusters
        min_avg_dist = min(min_avg_dist, avg_cluster_distance)

        random_starts -= 1

    print("Initial Centroid Assignments: ", len(centroid_assignments), ", Same Assignment Skipped Count: ", count)
    # print("Centroids: ", centroid_assignments)
    print("min_avg_dist: ", min_avg_dist, ", random_starts:", random_starts)
    return min_clusters


def k_means(matrix, k: int, max_iters: int):
    # initialized cluster centroids with random points from the similarity matrix
    k_centers = initialize_cluster_centroids(k, matrix)
    # k_centers = randomly_initialize_cluster_centroids(k, matrix)

    # initialized a dictionary of set of points belonging to a cluster: k -> cluster_index, v -> set
    clusters = defaultdict(set)

    # iterative cluster update loop
    while max_iters > 0:

        # saves the current cluster assignments
        old_clusters = clusters
        clusters = defaultdict(set)

        # loop through every data point in the similarity matrix
        for data_idx, vector in enumerate(matrix):

            # initialize minimum distance and cluster index for the current point
            min_cluster_idx, min_cluster_distance = -1, float("inf")

            # compare distances of current point with centroid of each cluster
            for i, k in k_centers.items():
                dist = get_euclidean_distance(k, vector)
                min_cluster_idx = i if dist < min_cluster_distance else min_cluster_idx
                min_cluster_distance = min(min_cluster_distance, dist)

            # add the current point index to the closest cluster
            clusters[min_cluster_idx].add(data_idx)

        # end the loop if the clusters have converged
        if are_clusters_same(old_clusters, clusters):
            # print("are_clusters_same(old_clusters, clusters): ", True, ", max_iters: ", max_iters)
            break

        # compute the new cluster centroids
        k_centers = compute_cluster_centroids(clusters, matrix)
        max_iters -= 1

    return clusters


def print_clusters(clusters, columns):
    for k, v in sorted(clusters.items(), key=lambda kv: kv[0]):
        print("Cluster: " + str(k+1), ";", end=" ")
        li = [columns[index] for index in v]
        print("n_gestures: ", len(li), "; Gestures: ", li)


def pretty(d, indent=0):
    for key, value in sorted(d.items()):
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def task_4c(file_name, p, max_iterations, random_starts):
    sim_mat_df = pd.read_csv(file_name, index_col=0)
    sim_mat = sim_mat_df.to_numpy()

    # normalize the similarity matrix between 0 and 1
    # sim_mat = normalize_between_0_and_1(sim_mat)
    print("\n-----------------------------------------------------------\n")
    print("KMeans Clustering Results: ")
    print("\n-----------------------------------------------------------\n")
    print("---------------------MWDB K Means Clusters----------------")
    clusters = multiple_random_starts_k_means(sim_mat, p, max_iterations, random_starts)
    # clusters = k_means(sim_mat, p, max_iterations)
    # pretty(clusters, 2)
    print_clusters(clusters, sim_mat_df.columns)
    sklearn_kmeans(sim_mat, p, max_iterations, random_starts, sim_mat_df.columns)


def task_4d(file_name, p, max_iterations, random_starts):
    # Read sim matrix
    df = pd.read_csv(file_name)
    print("df.shape: ", df.shape)

    # Similarity Graph
    W = np.zeros((df.shape[0], df.shape[0]))

    for i in range(df.shape[0]):
        for j in range(1, df.shape[1]):
            if df.iloc[i, j] >= 0:
                W[i, j - 1] = df.iloc[i, j]

    D = np.diag(np.sum(W, axis=1))
    L = D - W

    w, v = np.linalg.eig(L)
    y = v[:, np.argsort(w)]
    y = y[:, : p]

    print("\n\n-----------------------------------------------------------\n")
    print("Spectral Clustering Results: ")
    print("\n-----------------------------------------------------------\n")
    print("-------------------MWDB Spectral Clusters-----------------")
    clusters = multiple_random_starts_k_means(y, p, max_iterations, random_starts)
    # pretty(clusters, 2)
    print_clusters(clusters, df.iloc[:, 0])
    sklearn_kmeans(y, p, max_iterations, random_starts, df.iloc[:, 0])


def sklearn_kmeans(sim_mat, p, max_iterations, random_starts, columns):
    kmeans = KMeans(n_clusters=p, n_init=random_starts, max_iter=max_iterations, verbose=0, random_state=2).fit(sim_mat)
    clusters = defaultdict(set)
    for ind, val in enumerate(kmeans.labels_):
        clusters[val].add(ind)

    print("\n--------------------SKLEARN Clusters----------------------")
    # pretty(clusters, 2)
    print_clusters(clusters, columns)
    # print("kmeans.n_iter_: ", kmeans.n_iter_, ", kmeans.inertia_: ", kmeans.inertia_)

def group_by_semantics(file_name, take_absolute=False):
    similarity_matrix = pd.read_csv(file_name, index_col=0)
    p_groups = {}
    for file_number, col in similarity_matrix.transpose().iterrows():
        maximum = float('-inf')
        index = 0
        for i, element in col.iteritems():
            if take_absolute:
                element = abs(element)
            if element > maximum:
                maximum = element
                index = i
        if index not in p_groups:
            p_groups[index] = set()
        p_groups[index].add(file_number)
    print_dict(p_groups)


def main():
    if len(sys.argv) < 3:
        print('Run python Task_1.py <Directory> <User Option> <p>')
        sys.exit(0)
    directory = sys.argv[1]
    user_option = sys.argv[2]
    p = int(sys.argv[3])

    if user_option == '1':
        file_name = "task3_dot_sim_matrix.csv"
    elif user_option == '2':
        file_name = "task3_pca_sim_matrix.csv"
    elif user_option == '3':
        file_name = "task3_svd_sim_matrix.csv"
    elif user_option == '4':
        file_name = "task3_nmf_sim_matrix.csv"
    elif user_option == '5':
        file_name = "task3_lda_sim_matrix.csv"
    elif user_option == '6':
        file_name = "task3_Edit_Dist_sim_mat.csv"
    elif user_option == '7':
        file_name = "task3_DTW_sim_matrix.csv"

    # Task 4a
    print("\n-----------------------------------------------------------\n")
    print("Task 4a Results: ")
    print("\n-----------------------------------------------------------\n")
    group_by_semantics('phase_3_task3_SVD_output.csv', take_absolute=True)

    # Task 4b
    print("\n-----------------------------------------------------------\n")
    print("Task 4b Results: ")
    print("\n-----------------------------------------------------------\n")
    group_by_semantics('phase_3_task3_NMF_output.csv', take_absolute=True)

    # change number of random starts to get better results
    # Add random_starts as an optional argument to the task
    max_iterations, random_starts = 100, 10000

    # K Means
    task_4c(file_name, p, max_iterations, random_starts)

    # reset the variable centroid assignments before running task 4d
    global centroid_assignments
    centroid_assignments = set()

    # Spectral
    task_4d(file_name, p, max_iterations, random_starts)


# works only for similarity matrices which do not contain NAN or inf
if __name__ == "__main__":
    main()
