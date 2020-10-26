import os
import pickle
from collections import Counter
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import sys
import json
import glob
import math

vector_model_data = {}
the_matrix = {}
flattening_map = []

# returns list of vector files based on vector model
def get_list_of_files(dir, vector_model):
    list_of_files = os.listdir(dir)
    return [file for file in list_of_files if file.__contains__(vector_model+'_vectors')]

# stores vector model data globally
def read_file_data(list_of_files, dir):
    for each_file in  list_of_files:
        file_path = dir + "/" + each_file
        file_handler = open(file_path, 'rb')
        vector_model_data[each_file.split('.')[0].split('_')[-1]] = pickle.load(file_handler)

# return list of features across dimensions
def get_list_of_features():
    list_of_features = []
    for each_file in vector_model_data:
        for each_dimension in vector_model_data[each_file]:
            for each_sensor in vector_model_data[each_file][each_dimension]:
                for each_word in vector_model_data[each_file][each_dimension][each_sensor]:
                    list_of_features.append((each_dimension, each_sensor, each_word[0]))
    return list_of_features

# returns list of unique features
def get_unique_features(list_of_features):
    return list(Counter(list_of_features).keys())

# globally stores the feature matrix and creates a name map for each file
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

# returns similarity based on dot product of vectors
def get_dot_product_similarity(gesture_vector):
    list_of_similarities = []
    for each_file in the_matrix:
        score = 0
        for i in range(len(the_matrix[each_file])):
            score = score + (gesture_vector[i] * the_matrix[each_file][i])

        list_of_similarities.append((each_file, score))

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)

# returns similarity based on latent features retrieved from PCA
def get_pca_similarity(flattened_matrix, gesture_file, no_of_components):
    pca_gestures = PCA(no_of_components)
    transformed_matrix = pca_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_cosine_similarity(transformed_matrix, gesture_vector)

# returns similarity based on latent features retrieved from SVD
def get_svd_similarity(flattened_matrix, gesture_file, no_of_components):
    svd_gestures = TruncatedSVD(no_of_components)
    transformed_matrix = svd_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_cosine_similarity(transformed_matrix, gesture_vector)

# returns similarity based on latent features retrieved from NMF
def get_nmf_similarity(flattened_matrix, gesture_file, no_of_components):
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0, max_iter=4000)
    transformed_matrix = nmf_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_cosine_similarity(transformed_matrix, gesture_vector)

# returns similarity based on latent features retrieved from LDA
def get_lda_similarity(flattened_matrix, gesture_file, no_of_components):
    lda_gestures = LatentDirichletAllocation(n_components=no_of_components, random_state=0)
    transformed_matrix = lda_gestures.fit_transform(flattened_matrix)
    gesture_vector = transformed_matrix[flattening_map.index(gesture_file)]
    return get_kl_divergence(transformed_matrix, gesture_vector)

# returns cosine similarity between two vectors
def get_cosine_similarity(transformed_matrix, gesture_vector):
    list_of_similarities = []
    for i in range(len(transformed_matrix)):
        score = 0
        gesture_score = 0
        vector_score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * transformed_matrix[i][j])
            gesture_score += gesture_vector[j] ** 2
            vector_score += transformed_matrix[i][j] ** 2
        final_score = score/(math.sqrt(gesture_score) * math.sqrt(vector_score))
        list_of_similarities.append((flattening_map[i], final_score))

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)

# returns kl divergence between two probability distributions
def get_kl_divergence(transformed_matrix, gesture_vector):
    list_of_similarities = []
    for i in range(len(transformed_matrix)):
        score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * np.log(gesture_vector[j]/transformed_matrix[i][j]))
        
        final_score = 1 / (1 + score)
        list_of_similarities.append((flattening_map[i], final_score))
    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)

# flattens matrix for decompositions and creates a map for indices to file names
def flatten_the_matrix():
    temp_matrix = []
    for each_file in the_matrix:
        flattening_map.append(each_file)
        temp_matrix.append(the_matrix[each_file])
    return np.array(temp_matrix)

# returns the dtw cost between two vectors
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

# returns the edit distance cost between two files
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

# prints output in legible format
def print_results(result):
    for r in result:
        print(r)

if __name__ == '__main__':

    if len(sys.argv) < 5:
        print('Run python task2.py <Directory> <Gesture File> <Vector Model> <User Option> <k>')
        sys.exit(0)
    dir = sys.argv[1]
    gesture_file = sys.argv[2]
    vector_model = sys.argv[3]
    user_option = sys.argv[4]
    k = int(sys.argv[5])

    print("Directory: {}\nGesture File: {}\nVector Model: {}\nUser Option: {}\nk: {}".format(dir, gesture_file, vector_model, user_option, k))

    # function calls to initialize all global variables
    list_of_files = get_list_of_files(dir, vector_model)
    read_file_data(list_of_files, dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)
    gesture_vector = the_matrix[gesture_file]

    if user_option == '1':
        result = get_dot_product_similarity(gesture_vector)[:10]
        print("Printing dot product result")
        print_results(result)
    else:
        flattened_matrix = flatten_the_matrix()

        if user_option == '2':
            result = get_pca_similarity(flattened_matrix, gesture_file, k)[:10]
            print("Printing the pca similarity results")
            print_results(result)
        elif user_option == '3':
            result = get_svd_similarity(flattened_matrix, gesture_file, k)[:10]
            print("Printing the svd similarity results")
            print_results(result)
        elif user_option == '4':
            result = get_nmf_similarity(flattened_matrix, gesture_file, k)[:10]
            print("Printing the nmf similarity results")
            print_results(result)
        elif user_option == '5':
            result = get_lda_similarity(flattened_matrix, gesture_file, k)[:10]
            print("Printing the lda similarity results")
            print_results(result)
        elif user_option == '6':
            fnames = glob.glob("./" + dir + "/*.wrds")
            fnames.sort()

            qword = json.load(open('./' + dir + '/' + gesture_file + '.wrds'))  # .wrd
            comp = list(qword.keys())

            sim = []
            for gfile in fnames:
                f = os.path.splitext(os.path.basename(gfile))[0]
                fword = json.load(open('./' + dir + '/' + f + '.wrds'))  # .wrd

                temp = []
                for c in comp:
                    for senid in fword[c]:  # for wrd files
                        wf = list(np.array(fword[c][str(senid)]['words'])[:, 0])
                        wq = list(np.array(qword[c][str(senid)]['words'])[:, 0])
                        temp.append(editdist(wf, wq))

                sim.append((f, 1 / (1 + np.average(temp))))

            sim.sort(key=lambda x: x[1], reverse=True)
            print("Printing the edit distance similarity results")
            print(np.array(sim)[:10])

        elif user_option == '7':
            fnames = glob.glob("./" + dir + "/*.wrds")
            fnames.sort()

            qword = json.load(open('./' + dir + '/' + gesture_file + '.wrds'))  # .wrd
            comp = list(qword.keys())

            sim = []
            for gfile in fnames:
                f = os.path.splitext(os.path.basename(gfile))[0]
                fword = json.load(open('./' + dir + '/' + f + '.wrds'))  # .wrd

                temp = []
                for c in comp:
                    for senid in fword[c]:  # for wrd files
                        # wrd files
                        wf = list(np.array(fword[c][str(senid)]['words'])[:, 0])
                        wf_c = list(np.array(fword[c][str(senid)]['words'])[:, 1])
                        wq = list(np.array(qword[c][str(senid)]['words'])[:, 0])
                        wq_c = list(np.array(qword[c][str(senid)]['words'])[:, 1])
                        temp.append(dtw(wf, wq, wf_c, wq_c))

                sim.append((f, 1 / (1 + np.average(temp))))

            sim.sort(key=lambda x: x[1], reverse=True)
            print("Printing the dtw similarity results")
            print(np.array(sim)[:10])