import math
import os
import pickle
from collections import Counter

import glob
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from main import phase2_EDIT_Dist as edit_distance
from main import phase2_DTW as dtw

vector_model_data = {}
the_matrix = {}
flattening_map = []


def get_list_of_files(dir, vector_model):
    list_of_files = os.listdir(dir)
    return [file for file in list_of_files if file.__contains__(vector_model + '_vectors')]


def read_file_data(list_of_files, dir):
    for each_file in list_of_files:
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
            if (each_feature in word_list):
                index = word_list.index(each_feature)
                temp_list.append(value_list[index])
            else:
                temp_list.append(0)

        the_matrix[each_file] = temp_list


def get_orthonormal_dot(transformed_matrix, gesture_vector):
    list_of_similarities = {}
    for i in range(len(transformed_matrix)):
        score = 0
        # curr_score = 0
        # gest_score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * transformed_matrix[i][j])
            # curr_score += transformed_matrix[i][j] ** 2
            # gest_score += gesture_vector[j] ** 2
        list_of_similarities[flattening_map[i]] = score
        # list_of_similarities[flattening_map[i]] = score / (math.sqrt(curr_score) * math.sqrt(gest_score))
    return list_of_similarities


def flatten_the_matrix():
    temp_matrix = []
    for each_file in the_matrix:
        flattening_map.append(each_file)
        temp_matrix.append(the_matrix[each_file])
    return np.array(temp_matrix)


def get_dot_product_similarity_matrix():
    similarity_matrix = {}
    for gesture_file in the_matrix:
        gesture_vector = the_matrix[gesture_file]
        list_of_similarities = {}
        for each_file in the_matrix:
            score = 0
            for i in range(len(the_matrix[each_file])):
                score = score + (gesture_vector[i] * the_matrix[each_file][i])
            list_of_similarities[each_file] = score
        similarity_matrix[gesture_file] = list_of_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_dot_sim_matrix.csv')
    return s.to_numpy()


def get_pca_similarity_matrix(flattened_matrix, no_of_components):
    similarity_matrix = {}
    pca_gestures = PCA(no_of_components)
    transformed_matrix = pca_gestures.fit_transform(flattened_matrix)

    for gesture_file in the_matrix:
        gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_orthonormal_dot(transformed_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        print(gesture_file, x)
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_pca_sim_matrix.csv')
    return s.to_numpy()


def get_svd_similarity_matrix(flattened_matrix, no_of_components):
    similarity_matrix = {}
    svd_gestures = TruncatedSVD(no_of_components)
    transformed_matrix = svd_gestures.fit_transform(flattened_matrix)

    for gesture_file in the_matrix:
        gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_orthonormal_dot(transformed_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        print(gesture_file, x)
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_svd_sim_matrix.csv')
    return s.to_numpy()


def get_nmf_similarity_matrix(flattened_matrix, no_of_components):
    similarity_matrix = {}
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0, max_iter=4000)
    transformed_matrix = nmf_gestures.fit_transform(flattened_matrix)

    for gesture_file in the_matrix:
        gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_orthonormal_dot(transformed_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        print(gesture_file, x)
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_nmf_sim_matrix.csv')
    return s.to_numpy()


def get_lda_similarity_matrix(flattened_matrix, no_of_components):
    similarity_matrix = {}
    lda_gestures = LatentDirichletAllocation(n_components=no_of_components, random_state=0)
    transformed_matrix = lda_gestures.fit_transform(flattened_matrix)

    for gesture_file in the_matrix:
        gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_orthonormal_dot(transformed_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        print(gesture_file, x)
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_lda_sim_matrix.csv')
    return s.to_numpy()


def get_SVD_components(no_of_components, similarity_matrix):
    svd_gestures = TruncatedSVD(no_of_components)
    svd_gestures.fit_transform(similarity_matrix)
    print("#####################################################")
    print("Printing various stats for SVD")
    print("n_components_", svd_gestures.n_components)
    print("n_features_", svd_gestures.n_features_in_)
    print("components_", svd_gestures.components_)
    print("explained_variance_", svd_gestures.explained_variance_)
    print("explained_variance_ratio_", svd_gestures.explained_variance_ratio_)
    print("singular_values_", svd_gestures.singular_values_)


def get_NMF_components(no_of_components, similarity_matrix):
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0)
    nmf_gestures.fit_transform(similarity_matrix)
    print("#####################################################")
    print("Printing various stats for NMF")
    print("nmf_gestures.components_", len(nmf_gestures.components_))
    print("nmf_gestures.components_", np.sum(np.array(nmf_gestures.components_[0])))
    print("nmf_gestures.n_components", nmf_gestures.n_components)
    print("nmf_gestures.n_features_in_", nmf_gestures.n_features_in_)
    print("nmf_gestures.components_", nmf_gestures.components_)
    print("nmf_gestures.l1_ratio", nmf_gestures.l1_ratio)
    print("nmf_gestures.n_components_", nmf_gestures.n_components_)


def get_DTW_similarity_matrix():
    data_dir = 'data'

    file_names = glob.glob("./" + data_dir + "/*.wrd")
    file_names.sort()
    for i in range(len(file_names)):
        file_names[i] = os.path.splitext(os.path.basename(file_names[i]))[0]

    df = pd.DataFrame(0.0, index=file_names, columns=file_names)
    for i in range(len(file_names)):
        for j in range(i, len(file_names)):
            f1 = json.load(open('./' + data_dir + '/' + file_names[i] + '.wrd'))
            f2 = json.load(open('./' + data_dir + '/' + file_names[j] + '.wrd'))
            comp = list(f1.keys())

            temp = []
            for c in comp:
                for sensor_id in f1[c]:
                    w1 = list(np.array(f1[c][str(sensor_id)]['words'])[:, 0])
                    w1_c = list(np.array(f1[c][str(sensor_id)]['words'])[:, 1])
                    w2 = list(np.array(f2[c][str(sensor_id)]['words'])[:, 0])
                    w2_c = list(np.array(f2[c][str(sensor_id)]['words'])[:, 1])
                    dtw_val = dtw.dtw(w1, w2, w1_c, w2_c)
                    temp.append(1 / (1 + dtw_val))

            f1 = file_names[i]
            f2 = file_names[j]

            average = np.average(temp)

            df[f1][f2] = average
            df[f2][f1] = average

    df.to_csv('task3_DTW_sim_matrix.csv')
    print("dtw similarity matrix", df)
    return df


def get_ED_similarity_matrix():
    # Task 3a Part 1 user option 6:
    data_dir = 'data'
    file_names = glob.glob("./" + data_dir + "/*.wrd")
    file_names.sort()
    for i in range(len(file_names)):
        file_names[i] = os.path.splitext(os.path.basename(file_names[i]))[0]

    df = pd.DataFrame(0.0, index=file_names, columns=file_names)

    for i in range(len(file_names)):
        for j in range(i, len(file_names)):
            f1 = json.load(open('./' + data_dir + '/' + file_names[i] + '.wrd'))
            f2 = json.load(open('./' + data_dir + '/' + file_names[j] + '.wrd'))
            comp = list(f1.keys())

            temp = []
            for c in comp:
                for sensor_id in f1[c]:
                    w1 = list(np.array(f1[c][str(sensor_id)]['words'])[:, 0])
                    w2 = list(np.array(f2[c][str(sensor_id)]['words'])[:, 0])
                    temp.append(edit_distance.editdist(w1, w2))

            f1 = file_names[i]
            f2 = file_names[j]

            for k in range(len(temp)):
                temp[k] = 1 / (1 + temp[k])

            df[f1][f2] = np.average(temp)
            df[f2][f1] = np.average(temp)

    df.to_csv('task3_Edit_Dist_sim_mat.csv')
    print("edit distance similarity matrix", df)
    return df


if __name__ == '__main__':
    data_dir = "./data"
    vector_model = "tf"
    k = 50
    p = 3
    # TODO Should p and k be constants or inputs from the user?
    # k = int(input('Enter k: \t'))
    # p = int(input('Enter p: \t'))

    list_of_files = get_list_of_files(data_dir, vector_model)
    read_file_data(list_of_files, data_dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)

    sim_matrix = None
    user_option = input('1. Dot product | [2,3,4,5]. [PCA,SVD,NMF,LDA] | 6. Edit dist | 7. DTW :\n')

    if user_option == '1':
        sim_matrix = get_dot_product_similarity_matrix()
    else:
        flattened_matrix = flatten_the_matrix()

        if user_option == '2':
            sim_matrix = get_pca_similarity_matrix(flattened_matrix, k)
        elif user_option == '3':
            sim_matrix = get_svd_similarity_matrix(flattened_matrix, k)
        elif user_option == '4':
            sim_matrix = get_nmf_similarity_matrix(flattened_matrix, k)
        elif user_option == '5':
            sim_matrix = get_lda_similarity_matrix(flattened_matrix, k)
        elif user_option == '6':
            sim_matrix = get_ED_similarity_matrix()
        elif user_option == '7':
            sim_matrix = get_DTW_similarity_matrix()

    get_SVD_components(p, sim_matrix)
    # currently, NMF is giving out error only with PCA results. So for PCA results we are adding the minimum most
    # element(and only if it is negative) to all the elements in the array

    # compute user option here
    # If the negative error occurs with any other algorithm, add the corresponding user option to set here
    if user_option in set('2'):
        min_entry = math.fabs(np.amin(sim_matrix))
        row_count = sim_matrix.shape[0]
        col_count = sim_matrix.shape[1]
        if min_entry < 0:
            for i in range(row_count):
                for j in range(col_count):
                    sim_matrix[i][j] += min_entry
    get_NMF_components(p, sim_matrix)
