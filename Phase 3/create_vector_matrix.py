import os
import pickle
from collections import Counter
import json
import sys

vector_model_data = {}
the_matrix = []
the_matrix_name_map = []


# returns list of vector files based on vector model
def get_list_of_files(v_dir, vector_model):
    list_of_files = os.listdir(v_dir)
    return [file for file in list_of_files if file.__contains__(vector_model + '_vectors')]


# stores vector model data globally
def read_file_data(list_of_files, v_dir):
    for each_file in list_of_files:
        file_path = v_dir + "/" + each_file
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
            if (each_feature in word_list):
                index = word_list.index(each_feature)
                temp_list.append(value_list[index])
            else:
                temp_list.append(0)

        the_matrix.append(temp_list)
        the_matrix_name_map.append(each_file)


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def initialize():
    global vector_model_data, the_matrix, the_matrix_name_map
    vector_model_data = {}
    the_matrix = []
    the_matrix_name_map = []


def generate_vector_matrix(v_dir):
    # v_dir = input("Enter directory of vectors: ")
    # model = input("Enter vector model - tf or tfidf: ")
    models = ['tf', 'tfidf']

    # v_dir, model = "3_class_gesture_data", "tf"  # default values
    for model in models:
        initialize()
        file_list = get_list_of_files(v_dir, model)
        read_file_data(file_list, v_dir)
        list_of_features = get_list_of_features()
        set_of_features = get_unique_features(list_of_features)
        form_the_matrix(set_of_features)

        with open(model + '_feature_list.pkl', 'wb') as f:
            pickle.dump(set_of_features, f)

        output_vectors = {}

        for (i, j) in zip(the_matrix_name_map, the_matrix):
            output_vectors[i] = j

        with open(model+'_vectors.json', 'w') as fp:
            json.dump(output_vectors, fp)


if __name__ == '__main__':

    if len(sys.argv) < 1:
        print('Run python create_vector_matrix.py <Directory>')
        sys.exit(0)
    user_dir = sys.argv[1]
    # user_dir = input("Enter directory of vectors: ")
    generate_vector_matrix(user_dir)
