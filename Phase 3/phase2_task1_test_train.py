import os
import pickle
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import sys
import json

vector_model_data = {}
the_matrix = []
the_matrix_name_map = []
map_of_words = []
output_data = {}


# returns list of vector files based on vector model
def get_list_of_files(dir, vector_model, data_type):
    list_of_files = os.listdir(dir)
    if data_type == 'train':
        return [file for file in list_of_files if
                (file.__contains__(vector_model + '_vectors') and file.count("_") == 2)]
    if data_type == 'test':
        return [file for file in list_of_files if
                (file.__contains__(vector_model + '_vectors') and file.count("_") == 3)]


# stores vector model data globally
def read_file_data(list_of_files, dir):
    for each_file in list_of_files:
        file_path = dir + "/" + each_file
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

        the_matrix.append(temp_list)
        the_matrix_name_map.append(each_file)


# performs PCA on flattened matrix and writes output as per project requirement
def get_PCA_components(no_of_components, dir, data_type):
    print("Performing PCA on flattented matrix")
    flattened_matrix = np.array(the_matrix)
    pca_gestures = PCA(no_of_components)
    pca_gestures.fit_transform(flattened_matrix)
    pca_gestures.score(flattened_matrix)
    get_the_output(pca_gestures, dir)
    write_transformed_matrix(pca_gestures, dir, flattened_matrix, data_type, "PCA")


# performs SVD on flattened matrix and writes output as per project requirement
def get_SVD_components(no_of_components, dir, data_type):
    print("Performing SVD on flattened matrix")
    flattened_matrix = np.array(the_matrix)
    svd_gestures = TruncatedSVD(no_of_components)
    svd_gestures.fit_transform(flattened_matrix)
    get_the_output(svd_gestures, dir)
    write_transformed_matrix(svd_gestures, dir, flattened_matrix, data_type, "SVD")


# performs NMF on flattened matrix and writes output as per project requirement
def get_NMF_components(no_of_components, dir, data_type):
    print("Performing NMF on flattened matrix")
    flattened_matrix = np.array(the_matrix)
    nmf_gestures = NMF(n_components=no_of_components, init='random', random_state=0)
    nmf_gestures.fit_transform(flattened_matrix)
    get_the_output(nmf_gestures, dir)
    write_transformed_matrix(nmf_gestures, dir, flattened_matrix, data_type, "NMF")


# performs LDA on flattened matrix and writes output as per project requirement
def get_LDA_components(no_of_components, dir, data_type):
    print("Performing LDA on flattened matrix")
    flattened_matrix = np.array(the_matrix)
    lda_gestures = LatentDirichletAllocation(n_components=no_of_components, random_state=0)
    lda_gestures.fit_transform(flattened_matrix)
    get_the_output(lda_gestures, dir)
    write_transformed_matrix(lda_gestures, dir, flattened_matrix, data_type, "LDA")


# stores the output in descending order of contribution scores
def get_the_output(transformed_object, dir):
    print("Storing the output after decomposition")
    component_matrix = transformed_object.components_
    latent_semantic_number = 1
    for each_latent_semantic in component_matrix:
        temp_list = []
        for i in range(len(each_latent_semantic)):
            temp_list.append((map_of_words[i], each_latent_semantic[i]))

        output_data[str(latent_semantic_number)] = sorted(temp_list, key=lambda x: x[1], reverse=True)
        latent_semantic_number = latent_semantic_number + 1

    file_name = "phase_2_task_1_output"
    outF = open(os.path.join(file_name + ".json"), "w")
    json.dump(output_data, outF, default=convert)
    outF.close()


# stores the transformed matrix as metadata
def write_transformed_matrix(transformed_object, dir, flattened_matrix, data_type, model):
    print("Storing the transformed matrix after decomposition")
    file_name = 'phase2_task1_transformed_matrix_u'
    if data_type == "train":
        file_name = "phase2_task1_transformed_matrix_train"
    if data_type == "test":
        file_name = "phase2_task1_transformed_matrix_test"

    transformed_matrix = transformed_object.fit_transform(flattened_matrix)

    output_dictionary = {}
    index = 0
    for each_file in the_matrix_name_map:
        output_dictionary[each_file] = transformed_matrix[index].tolist()
        index = index + 1

    outF = open(os.path.join(file_name + "_" + model + ".json"), "w")
    json.dump(output_dictionary, outF, default=convert)
    outF.close()


# returns the integer version of a numpy integer
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print('Run python phase2_task1.py <Directory> <Vector Model> <User Option> <k> <test/train>')
        sys.exit(0)
    directory = sys.argv[1]
    vector_model = sys.argv[2]
    user_option = int(sys.argv[3])
    k = int(sys.argv[4])
    data_type = sys.argv[5]

    print("Directory: {}\nVector Model: {}\nUser Option: {}\nk: {}\ntrain/test: {}".format(directory, vector_model,
                                                                                         user_option, k, data_type))

    if data_type == 'train':
        # function calls to initialize all global variables
        list_of_files = get_list_of_files(directory, vector_model, data_type)
        read_file_data(list_of_files, directory)
        list_of_features = get_list_of_features()
        set_of_features = get_unique_features(list_of_features)
        map_of_words = set_of_features
        form_the_matrix(set_of_features)

        if user_option == 1:
            get_PCA_components(k, directory, data_type)
        if user_option == 2:
            get_SVD_components(k, directory, data_type)
        if user_option == 3:
            get_NMF_components(k, directory, data_type)
        if user_option == 4:
            get_LDA_components(k, directory, data_type)

    if data_type == 'test':
        # function calls to initialize all global variables
        list_of_files = get_list_of_files(directory, vector_model, data_type)
        read_file_data(list_of_files, directory)
        list_of_features = get_list_of_features()
        set_of_features = get_unique_features(list_of_features)
        map_of_words = set_of_features
        form_the_matrix(set_of_features)

        if user_option == 1:
            get_PCA_components(k, directory, data_type)
        if user_option == 2:
            get_SVD_components(k, directory, data_type)
        if user_option == 3:
            get_NMF_components(k, directory, data_type)
        if user_option == 4:
            get_LDA_components(k, directory, data_type)

    pass
