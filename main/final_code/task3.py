import math
import os
import pickle
from collections import Counter
import sys
import glob
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

vector_model_data = {}
the_matrix = {}
flattening_map = []
output_data = {}


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
        curr_score = 0
        gest_score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * transformed_matrix[i][j])
            curr_score += transformed_matrix[i][j] ** 2
            gest_score += gesture_vector[j] ** 2
        # list_of_similarities[flattening_map[i]] = score
        final_score = score / (math.sqrt(curr_score) * math.sqrt(gest_score))
        list_of_similarities[flattening_map[i]] = final_score
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
    print('PCA, k= ', no_of_components)
    similarity_matrix = {}
    pca_gestures = PCA(no_of_components)
    transformed_matrix = pca_gestures.fit_transform(flattened_matrix)

    for gesture_file in the_matrix:
        gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_orthonormal_dot(transformed_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        # print(gesture_file, x)
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
        # print(gesture_file, x)
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
        # print(gesture_file, x)
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
        # print(gesture_file, x)
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    s.to_csv('task3_lda_sim_matrix.csv')
    return s.to_numpy()


def get_SVD_components(no_of_components, similarity_matrix):
    print("SVD, p=", no_of_components)
    svd_gestures = TruncatedSVD(no_of_components)
    svd_gestures.fit_transform(similarity_matrix)
    get_the_output(svd_gestures, data_dir, "SVD")

def get_NMF_components(no_of_components, similarity_matrix, data_dir):
    print("NMF, p=", no_of_components)
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0)
    nmf_gestures.fit_transform(similarity_matrix)
    get_the_output(nmf_gestures, data_dir, "NMF")

def dtw(vector1, vector2, cost1, cost2):
    assert len(vector1) == len(cost1)
    assert len(vector2) == len(cost2)
    row = len(vector1) + 1
    col = len(vector2) + 1
    cost1.reverse()
    dp = [[float('inf')] * col for i in range(row)]
    dp[row - 1][0] = 0
    for i in range(row - 2, -1, -1):
        for j in range(1, col):
            cost = 0
            if vector1[i] != vector2[j - 1]:
                cost = abs(cost1[i] - cost2[j - 1])
            dp[i][j] = cost + min(dp[i + 1][j], dp[i][j - 1], dp[i + 1][j - 1])

    return dp[0][-1]

def get_DTW_similarity_matrix(data_dir):

    file_names = glob.glob("./" + data_dir + "/*.wrds")
    file_names.sort()
    for i in range(len(file_names)):
        file_names[i] = os.path.splitext(os.path.basename(file_names[i]))[0]

    df = pd.DataFrame(0.0, index=file_names, columns=file_names)
    for i in range(len(file_names)):
        for j in range(i, len(file_names)):
            f1 = json.load(open('./' + data_dir + '/' + file_names[i] + '.wrds'))
            f2 = json.load(open('./' + data_dir + '/' + file_names[j] + '.wrds'))
            comp = list(f1.keys())

            temp = []
            for c in comp:
                for sensor_id in f1[c]:
                    w1 = list(np.array(f1[c][str(sensor_id)]['words'])[:, 0])
                    w1_c = list(np.array(f1[c][str(sensor_id)]['words'])[:, 1])
                    w2 = list(np.array(f2[c][str(sensor_id)]['words'])[:, 0])
                    w2_c = list(np.array(f2[c][str(sensor_id)]['words'])[:, 1])
                    dtw_val = dtw(w1, w2, w1_c, w2_c)
                    temp.append(1 / (1 + dtw_val))

            f1 = file_names[i]
            f2 = file_names[j]

            average = np.average(temp)

            df[f1][f2] = average
            df[f2][f1] = average

    df.to_csv('task3_DTW_sim_matrix.csv')
    print("dtw similarity matrix", df)
    return df

def editdist(s, t):  # for wrd files
    rows = len(s) + 1
    cols = len(t) + 1

    dist = [[0 for x in range(cols)] for x in range(rows)]

    for row in range(1, rows):
        dist[row][0] = row * 3

    for col in range(1, cols):
        dist[0][col] = col * 3

    for row in range(1, rows):
        for col in range(1, cols):
            if s[row - 1] == t[col - 1]:
                cost = 0
            else:
                cost = 0
                for i in range(len(s[row - 1])):
                    if s[row - 1][i] != t[col - 1][i]:
                        cost += 1

            dist[row][col] = min(dist[row - 1][col] + 3,  # deletes
                                 dist[row][col - 1] + 3,  # inserts
                                 dist[row - 1][col - 1] + cost)  # substitution
    return dist[row][col]


def get_ED_similarity_matrix(data_dir, k):
    # Task 3a Part 1 user option 6:
    file_names = glob.glob("./" + data_dir + "/*.wrd")
    file_names.sort()
    for i in range(len(file_names)):
        file_names[i] = os.path.splitext(os.path.basename(file_names[i]))[0]

    df = pd.DataFrame(0.0, index=file_names, columns=file_names)

    for i in range(len(file_names)):
        for j in range(i, len(file_names)):
            f1 = json.load(open('./' + data_dir + '/' + file_names[i] + '.wrds'))
            f2 = json.load(open('./' + data_dir + '/' + file_names[j] + '.wrds'))
            comp = list(f1.keys())

            temp = []
            for c in comp:
                for sensor_id in f1[c]:
                    w1 = list(np.array(f1[c][str(sensor_id)]['words'])[:, 0])
                    w2 = list(np.array(f2[c][str(sensor_id)]['words'])[:, 0])
                    temp.append(editdist(w1, w2))

            f1 = file_names[i]
            f2 = file_names[j]

            for k in range(len(temp)):
                temp[k] = 1 / (1 + temp[k])

            df[f1][f2] = np.average(temp)
            df[f2][f1] = np.average(temp)

    df.to_csv('task3_Edit_Dist_sim_mat.csv')
    print("edit distance similarity matrix", df)
    return df

def get_the_output(transformed_object, dir, type):
    component_matrix = transformed_object.components_
    latent_semantic_number = 1
    for each_latent_semantic in component_matrix:
        temp_list = {}
        for i in range(len(each_latent_semantic)):
            temp_list[flattening_map[i]] = each_latent_semantic[i]
        key = str(latent_semantic_number)
        output_data[key] = temp_list
        print("Printing the type of output_data_entry ", key)
        latent_semantic_number = latent_semantic_number + 1

    file_name = "phase_2_task3_" + type + "_contributions.csv"
    semantic_contributions = pd.DataFrame.from_dict(output_data, orient='index')
    semantic_contributions.to_csv(file_name)

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print('Run python task3.py <Directory> <Vector Model> <User Option> <p> <k>')
        sys.exit(0)
    data_dir = sys.argv[1]
    vector_model = sys.argv[2]
    user_option = sys.argv[3]
    k = int(sys.argv[4])
    p = int(sys.argv[5])

    print("Directory: {}\nVector Model: {}\nUser Option: {}\np: {}\nk: {}\n".format(data_dir, vector_model, user_option, p, k))

    list_of_files = get_list_of_files(data_dir, vector_model)
    read_file_data(list_of_files, data_dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)

    sim_matrix = None

    flattened_matrix = flatten_the_matrix()
    if user_option == '1':
        sim_matrix = get_dot_product_similarity_matrix()
    elif user_option == '2':
        sim_matrix = get_pca_similarity_matrix(flattened_matrix, k)
    elif user_option == '3':
        sim_matrix = get_svd_similarity_matrix(flattened_matrix, k)
    elif user_option == '4':
        sim_matrix = get_nmf_similarity_matrix(flattened_matrix, k)
    elif user_option == '5':
        sim_matrix = get_lda_similarity_matrix(flattened_matrix, k)
    elif user_option == '6':
        sim_matrix = get_ED_similarity_matrix(data_dir, k)
    elif user_option == '7':
        sim_matrix = get_DTW_similarity_matrix(data_dir)

    # Performing latent semantic analysis with SVD
    get_SVD_components(p, sim_matrix)

    # Performing latent semantic analysis with NMF
    # currently, NMF is giving out error only with PCA results. So for PCA results we are adding the minimum most
    # element(and only if it is negative) to all the elements in the array

    # compute user option here
     # If the negative error occurs with any other algorithm, add the corresponding user option to set here
    if user_option in set('2'):
        min_entry = np.amin(sim_matrix)
        row_count = sim_matrix.shape[0]
        col_count = sim_matrix.shape[1]
        if min_entry < 0:
            min_entry = math.fabs(min_entry)
            for i in range(row_count):
                for j in range(col_count):
                    sim_matrix[i][j] += min_entry
    get_NMF_components(p, sim_matrix, data_dir)