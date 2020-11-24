import math
import os
import pickle
from collections import Counter
from collections import defaultdict
import sys
import glob
import json
import pandas as pd
import numpy as np
# import copy
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib import colors

vector_model_data = {}
the_matrix = {}
flattening_map = []
output_data = {}

# returns list of vector files based on vector model
def get_list_of_files(dir, vector_model):
    list_of_files = os.listdir(dir)
    return [file for file in list_of_files if file.__contains__(vector_model + '_vectors')]

# stores vector model data globally
def read_file_data(list_of_files, dir):
    for each_file in list_of_files:
        file_path = dir + "/" + each_file
        file_handler = open(file_path, 'rb')
        c = '_'
        vector_model_data[c.join(each_file.split('.')[0].split('_')[2:])] = pickle.load(file_handler)
    # for each_file in list_of_files:
    #     file_path = dir + "/" + each_file
    #     file_handler = open(file_path, 'rb')
    #     vector_model_data[each_file.split('.')[0].split('_')[-1]] = pickle.load(file_handler)

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
            if (each_feature in word_list):
                index = word_list.index(each_feature)
                temp_list.append(value_list[index])
            else:
                temp_list.append(0)

        the_matrix[each_file] = temp_list

# returns cosine similarity between two vectors
def get_cosine_similarity(transformed_matrix, gesture_vector):
    list_of_similarities = {}
    for i in range(len(transformed_matrix)):
        score = 0
        curr_score = 0
        gest_score = 0
        for j in range(len(transformed_matrix[i])):
            score = score + (gesture_vector[j] * transformed_matrix[i][j])
            curr_score += transformed_matrix[i][j] ** 2
            gest_score += gesture_vector[j] ** 2
        final_score = score / (math.sqrt(curr_score) * math.sqrt(gest_score))
        list_of_similarities[flattening_map[i]] = final_score
    return list_of_similarities

def flatten_the_matrix():
    temp_matrix = []
    for each_file in the_matrix:
        flattening_map.append(each_file)
        temp_matrix.append(the_matrix[each_file])
    return np.array(temp_matrix)

def get_cosine_similarity_matrix(flattened_matrix):
    # Task 3a Part 1 user option 2:
    similarity_matrix = {}
    # transformed_matrix = flattened_matrix
    # pca_gestures = PCA(no_of_components)
    # transformed_matrix = pca_gestures.fit_transform(flattened_matrix)
    # print("Creating similarity matrix based on PCA")
    for gesture_file in the_matrix:
        gesture_vector = flattened_matrix[flattening_map.index(gesture_file)]
        gesture_similarities = get_cosine_similarity(flattened_matrix, gesture_vector)
        x = {k: v for k, v in sorted(gesture_similarities.items(), key=lambda item: item[1], reverse=True)}
        similarity_matrix[gesture_file] = gesture_similarities
    s = pd.DataFrame.from_dict(similarity_matrix, orient="index")
    print("Saving similarity matrix")
    s.to_csv('cosine_sim_matrix.csv')
    return s.to_numpy()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Run python task3.py <Directory> <Vector Model>')
        sys.exit(0)

    data_dir = sys.argv[1]
    vector_model = sys.argv[2]


    print("Directory: {}\nVector Model: {}\n".format(data_dir, vector_model))


    list_of_files = get_list_of_files(data_dir, vector_model)
    read_file_data(list_of_files, data_dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)

    sim_matrix = None

    flattened_matrix = flatten_the_matrix()

    sim_matrix = get_cosine_similarity_matrix(flattened_matrix)
