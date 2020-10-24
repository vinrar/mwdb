import os
import pickle
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation

vector_model_data = {}
the_matrix = {}
flattening_map = []

def get_list_of_files(dir, vector_model):
    list_of_files = os.listdir(dir)
    return [file for file in list_of_files if file.__contains__(vector_model+'_vectors')]

def read_file_data(list_of_files, dir):
    for each_file in  list_of_files:
        file_path = dir + "/" + each_file
        file_handler = open(file_path, 'rb')
        vector_model_data[each_file.split('.')[0].split('_')[-1]] = pickle.load(file_handler)


def get_list_of_features():
    list_of_features = []
    for each_file in vector_model_data:
        for each_dimension in vector_model_data[each_file]:
            for each_sensor in vector_model_data[each_file][each_dimension]:
                for each_word in vector_model_data[each_file][each_dimension][each_sensor]:
                    list_of_features.append((each_dimension, each_sensor, each_word[0]))
    return list_of_features

def get_unique_features(list_of_features):
    return list(Counter(list_of_features).keys())

def form_the_matrix(set_of_features):
    for each_file in vector_model_data:
        word_list = []
        value_list = []
        for each_dimension in vector_model_data[each_file]:
            for each_sensor in vector_model_data[each_file][each_dimension]:
                for each_word in vector_model_data[each_file][each_dimension][each_sensor]:
                    word_list.append((each_dimension, each_sensor, each_word[0]))
                    value_list.append(each_word[1])

        temp_list = []
        for each_feature in set_of_features:
            if(each_feature in word_list):
                index = word_list.index(each_feature)
                temp_list.append(value_list[index])
            else:
                temp_list.append(0)

        the_matrix[each_file] = temp_list

def get_dot_product_similarity(gesture_vector):
    list_of_similarities = []
    for each_file in the_matrix:
        score = 0
        for i in range(len(the_matrix[each_file])):
            score = score + (gesture_vector[i] * the_matrix[each_file][i])

        list_of_similarities.append((each_file, score))

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)

def get_pca_similarity(flattened_matrix, gesture_file, no_of_components):
    # print("printing stuff about flattened matrix ", len(flattened_matrix[0]))
    pca_gestures = PCA(no_of_components)
    transformed_matrix = pca_gestures.fit_transform(flattened_matrix)
    # print("printnig stuff about transfolrmed matrix", transformed_matrix[0])
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_orthonormal_dot(transformed_matrix, gesture_vector)

def get_svd_similarity(flattened_matrix, gesture_file, no_of_components):
    svd_gestures = TruncatedSVD(no_of_components)
    transformed_matrix = svd_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_orthonormal_dot(transformed_matrix, gesture_vector)

def get_nmf_similarity(flattened_matrix, gesture_file, no_of_components):
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0, max_iter=4000)
    transformed_matrix = nmf_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_orthonormal_dot(transformed_matrix, gesture_vector)

def get_lda_similarity(flattened_matrix, gesture_file, no_of_components):
    lda_gestures = LatentDirichletAllocation(n_components=no_of_components, random_state=0)
    transformed_matrix = lda_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_orthonormal_dot(transformed_matrix, gesture_vector)


def get_orthonormal_dot(transformed_matrix, gesture_vector):
    list_of_similarities = []
    for i in range(len(transformed_matrix)):
        score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * transformed_matrix[i][j])

        list_of_similarities.append((flattening_map[i], score))

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)

def flatten_the_matrix():
    temp_matrix = []
    for each_file in the_matrix:
        flattening_map.append(each_file)
        # print("prinntig each_file while flattening ", each_file)
        temp_matrix.append(the_matrix[each_file])
    return np.array(temp_matrix)


# def get_pca_similarity2(flattened_matrix, gesture_file, no_of_components):
#     pca_gestures = PCA(no_of_components)
#     pca_gestures.fit_transform(flattened_matrix)
#     gesture_vector = the_matrix[gesture_file]
#
#     gesture_vector = pca_gestures.transform(np.array(gesture_vector))
#
#     list_of_similarities = []
#     for each_file in the_matrix:
#         score = 0
#         current_vector = pca_gestures.transform(np.array(the_matrix[each_file]))
#         for i in range(len(current_vector)):
#             score = score + (gesture_vector[i] * current_vector[i])
#
#         list_of_similarities.append((each_file, score))
#
#     return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    dir = "/home/asim/Desktop/ankit_acad_temp/MWDB/Phase_2_stuff/Amey_task0a_wrdfiles"
    vector_model = "tf"
    k = 50
    gesture_file = "28"

    list_of_files = get_list_of_files(dir, vector_model)
    read_file_data(list_of_files, dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)
    gesture_vector = the_matrix[gesture_file]

    result = get_dot_product_similarity(gesture_vector)
    print("printing dot product result ", result)

    flattened_matrix = flatten_the_matrix()

    result = get_pca_similarity(flattened_matrix, gesture_file, k)
    print("priihintg the pca similarity results", result)

    # result = get_pca_similarity2(flattened_matrix, gesture_file, k)
    # print("priihintg the pca similarity2 results", result)

    result = get_svd_similarity(flattened_matrix, gesture_file, k)
    print("priihintg the svd similarity results", result)

    result = get_nmf_similarity(flattened_matrix, gesture_file, k)
    print("priihintg the nmf similarity results", result)

    result = get_lda_similarity(flattened_matrix, gesture_file, k)
    print("priihintg the lda similarity results", result)

