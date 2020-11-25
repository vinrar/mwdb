import json
import numpy as np
import os
import pickle
import sys
from collections import Counter
from sklearn.decomposition import PCA

vector_model_data = {}
the_matrix = []
the_matrix_name_map = []
map_of_words = []
output_data = {}
user_input_model = ""
vector_model = ""


# returns list of vector files based on vector model
def get_list_of_files(v_dir, v_model):
    list_of_files = os.listdir(v_dir)
    return [file for file in list_of_files if file.__contains__(v_model + '_vectors')]


# stores vector model data globally
def read_file_data(list_of_files, v_dir):
    for each_file in list_of_files:
        file_path = v_dir + "/" + each_file
        file_handler = open(file_path, 'rb')
        vector_model_data[each_file.split('.')[0].split('_', 2)[-1]] = pickle.load(file_handler)


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
            if each_feature in word_list:
                index = word_list.index(each_feature)
                temp_list.append(value_list[index])
            else:
                temp_list.append(0)

        the_matrix.append(temp_list)
        the_matrix_name_map.append(each_file)


# performs PCA on flattened matrix and writes output as per project requirement
def get_PCA_components(no_of_components, dir):
    print("Performing PCA on flattented matrix")
    flattened_matrix = np.array(the_matrix)
    pca_gestures = PCA(no_of_components)
    pca_gestures.fit_transform(flattened_matrix)
    pca_gestures.score(flattened_matrix)

    # get_the_output(pca_gestures, dir)
    write_transformed_matrix(pca_gestures, dir, flattened_matrix)


# stores the transformed matrix as metadata
def write_transformed_matrix(transformed_object, dir, flattened_matrix):
    print("Storing the transformed matrix after decomposition")
    file_name = user_input_model+"_transformed_" + vector_model + "_vectors"
    transformed_matrix = transformed_object.fit_transform(flattened_matrix)

    pickle.dump(transformed_matrix, open(vector_model+"_"+user_input_model+"_fit_object.pkl", "wb"))

    output_dictionary = {}
    index = 0
    for each_file in the_matrix_name_map:
        output_dictionary[each_file] = transformed_matrix[index].tolist()
        index = index + 1

    outF = open(os.path.join(file_name + ".json"), "w")
    json.dump(output_dictionary, outF, default=convert)
    outF.close()


# returns the integer version of a numpy integer
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def initialize():
    global vector_model_data, the_matrix, the_matrix_name_map, map_of_words, output_data
    vector_model_data = {}
    the_matrix = []
    the_matrix_name_map = []
    map_of_words = []
    output_data = {}


def generate_vector_matrix(v_dir):
    models = ['tf', 'tfidf']
    global vector_model, user_input_model, map_of_words

    for model in models:
        initialize()
        vector_model = model
        user_input_model = 'pca'

        file_list = get_list_of_files(v_dir, model)
        read_file_data(file_list, v_dir)
        list_of_features = get_list_of_features()
        set_of_features = get_unique_features(list_of_features)
        map_of_words = set_of_features
        form_the_matrix(set_of_features)
        get_PCA_components(10, v_dir)

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
    generate_vector_matrix(user_dir)
