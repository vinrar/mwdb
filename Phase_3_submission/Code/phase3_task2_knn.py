import math
import os
import pickle
import sys
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split

vector_model_data = {}
the_matrix = {}
flattening_map = []


# returns list of vector files based on vector model
def get_list_of_files(dir, vector_model):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_folder, dir)
    list_of_files = os.listdir(data_dir)
    return [file for file in list_of_files if file.__contains__(vector_model + '_vectors')]


# stores vector model data globally
def read_file_data(list_of_files, dir):
    current_folder = os.path.dirname(os.path.abspath(__file__))
    for each_file in list_of_files:
        # file_path = dir + "/" + each_file
        data_dir = os.path.join(current_folder, dir)
        file_path = os.path.join(data_dir, each_file)
        file_handler = open(file_path, 'rb')
        if each_file.count("_") == 3:
            keys = each_file.split('.')[0].split('_')
            key = keys[-2] + "_" + keys[-1]
        else:
            key = each_file.split('.')[0].split('_')[-1]
        vector_model_data[key] = pickle.load(file_handler)


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


# returns similarity based on dot product of vectors
def get_distance_similarity(gesture_vector):
    list_of_similarities = []
    for each_file in the_matrix:
        # compute the similarity for only the main vectors
        if each_file.count("_") == 0:
            distance = 0
            # computing Eucledian distance here
            for i in range(len(the_matrix[each_file])):
                distance = distance + (gesture_vector[i] * the_matrix[each_file][i])

            distance = math.sqrt(distance)
            list_of_similarities.append((each_file, distance))

    if not list_of_similarities:
        return list_of_similarities

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)[:k + 1]


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
        final_score = score / (math.sqrt(gesture_score) * math.sqrt(vector_score))
        list_of_similarities.append((flattening_map[i], final_score))

    return sorted(list_of_similarities, key=lambda x: x[1], reverse=True)


# flattens matrix for decompositions and creates a map for indices to file names
def flatten_the_matrix():
    temp_matrix = []
    for each_file in the_matrix:
        flattening_map.append(each_file)
        temp_matrix.append(the_matrix[each_file])
    return np.array(temp_matrix)


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


def perform_knn(similarity_matrix, k, test_key):
    gesture_1 = 'vattene'
    gesture_2 = 'Combinato'
    gesture_3 = "D'Accordo"
    config_map = {gesture_1: [1, 31], gesture_2: [249, 279], gesture_3: [559, 589]}
    result_map = {}
    for key in config_map:
        config_range = config_map[key]
        for i in range(config_range[0], config_range[1] + 1):
            result_map[str(i)] = key

    count_map = {}
    max_count = 0
    max_key = ''
    for i in range(0, k):
        top_result = similarity_matrix[i]
        if top_result[0] in result_map:
            key = result_map[top_result[0]]
        else:
            key = 'U'

        if key in count_map:
            count_map[key] += 1
        else:
            count_map[key] = 1

        if count_map[key] > max_count:
            max_count = count_map[key]
            max_key = key

    print('According to KNN the best group the key falls into: ' + max_key)
    print('Correct value of the key: ' + str(result_map[test_key]))
    print('Keys in group are in range: ' + str(config_map[max_key]))
    if result_map[test_key] == max_key:
        return 1
    return 0


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print('Run python phase3_task2_knn.py <Directory> <Vector Model> <k>')
        sys.exit(0)

    directory = sys.argv[1]
    vector_model = sys.argv[2]
    k = int(sys.argv[3])

    print("Directory: {}\nVector Model: {}\nk: {}".format(directory, vector_model, k))

    # function calls to initialize all global variables
    list_of_files = get_list_of_files(directory, vector_model)
    read_file_data(list_of_files, directory)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    form_the_matrix(set_of_features)

    print(type(the_matrix))
    keys = list(the_matrix.keys())
    train, test = train_test_split(keys, test_size=0.2)
    total_count = 0
    correct_count = 0
    # for test_key in test:
    for test_key in keys:
        if test_key.count("_") == 1:
            total_count += 1
            gesture_vector = the_matrix[test_key]
            similarity_matrix = get_distance_similarity(gesture_vector)
            # ignoring the first key
            if similarity_matrix:
                similarity_matrix = similarity_matrix[1:]
                print("Printing the similarity results based on dot product")
                print_results(similarity_matrix)
                correct_count += perform_knn(similarity_matrix, k, test_key.split("_")[0])

    print("Total count: " + str(total_count))
    print("Count: " + str(correct_count))
    print("Accuracy: " + str(correct_count / total_count))
