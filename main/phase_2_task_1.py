import os
import pickle
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


vector_model_data = {}
the_matrix = []
the_matrix_name_map = []
map_of_words = []
output_data = {}

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

        the_matrix.append(temp_list)
        the_matrix_name_map.append(each_file)

def get_PCA_components(no_of_components, dir):
    flattened_matrix = np.array(the_matrix)
    pca_gestures = PCA(no_of_components)
    pca_gestures.fit_transform(flattened_matrix)
    pca_gestures.score(flattened_matrix)
    get_the_output(pca_gestures, dir)
    write_transformed_matrix(pca_gestures, dir, flattened_matrix)
    # print("prining various stats for PCA")
    # print("n_components_", pca_gestures.n_components_)
    # print("n_features_", pca_gestures.n_features_)
    # print("components_", len(pca_gestures.components_[0]))
    # print("components_", len(pca_gestures.components_))
    # print("explained_variance_", pca_gestures.explained_variance_)
    # print("explained_variance_ratio_", pca_gestures.explained_variance_ratio_)
    # print("singular_values_", pca_gestures.singular_values_)

def get_SVD_components(no_of_components, dir):
    flattened_matrix = np.array(the_matrix)
    svd_gestures = TruncatedSVD(no_of_components)
    svd_gestures.fit_transform(flattened_matrix)
    get_the_output(svd_gestures, dir)
    # print("#####################################################")
    # print("prining various stats for PCA")
    # print("n_components_", svd_gestures.n_components)
    # print("n_features_", svd_gestures.n_features_in_)
    # print("components_", svd_gestures.components_)
    # print("explained_variance_", svd_gestures.explained_variance_)
    # print("explained_variance_ratio_", svd_gestures.explained_variance_ratio_)
    # print("singular_values_", svd_gestures.singular_values_)

def get_NMF_components(no_of_components, dir):
    flattened_matrix = np.array(the_matrix)
    nmf_gestures  = NMF(n_components=no_of_components, init='random', random_state=0)
    nmf_gestures.fit_transform(flattened_matrix)
    get_the_output(nmf_gestures, dir)
    # print("nmf_gestures.components_", len(nmf_gestures.components_))
    # print("nmf_gestures.components_", np.sum(np.array(nmf_gestures.components_[0])))
    # print("#####################################################")
    # print("nmf_gestures.n_components", nmf_gestures.n_components)
    # print("nmf_gestures.n_features_in_", nmf_gestures.n_features_in_)
    # print("nmf_gestures.components_", nmf_gestures.components_)
    # print("nmf_gestures.l1_ratio", nmf_gestures.l1_ratio)
    # print("nmf_gestures.n_components_", nmf_gestures.n_components_)

def get_LDA_components(no_of_components, dir):
    flattened_matrix = np.array(the_matrix)
    lda_gestures = LatentDirichletAllocation(n_components=no_of_components, random_state=0)
    lda_gestures.fit_transform(flattened_matrix)
    get_the_output(lda_gestures, dir)


def get_the_output(transformed_object, dir):
    component_matrix = transformed_object.components_
    latent_semantic_number = 1
    for each_latent_semantic in component_matrix:
        temp_list = []
        for i in range(len(each_latent_semantic)):
            temp_list.append((map_of_words[i], each_latent_semantic[i]))

        output_data[str(latent_semantic_number)] = sorted(temp_list, key=lambda x: x[1], reverse=True)
        # print("prinitng the type of output_data_entry ", type(output_data[str(latent_semantic_number)]))
        latent_semantic_number = latent_semantic_number + 1

    file_name = "phase_2_task_1_output"
    file_handler = open(dir + "/" + file_name, 'wb')
    pickle.dump(output_data, file_handler)

def write_transformed_matrix(transformed_object, dir, flattened_matrix):
    file_name = "phase_2_task_1_transformed_matrix"
    file_handler = open(dir + "/" + file_name, 'wb')
    transformed_matrix = transformed_object.fit_transform(flattened_matrix)

    output_dictionary = {}
    index = 0
    for each_file in the_matrix_name_map:
        output_dictionary[each_file] = transformed_matrix[index].tolist()
        index = index + 1
    pickle.dump(output_dictionary,  file_handler)


if __name__ == '__main__':
    dir = "/home/asim/Desktop/ankit_acad_temp/MWDB/Phase_2_stuff/Amey_task0a_wrdfiles"
    vector_model = "tf"
    k = 10

    list_of_files = get_list_of_files(dir, vector_model)
    read_file_data(list_of_files, dir)
    list_of_features = get_list_of_features()
    set_of_features = get_unique_features(list_of_features)
    map_of_words = set_of_features
    form_the_matrix(set_of_features)

    get_PCA_components(k, dir)
    # get_SVD_components(k, dir)
    # get_NMF_components(k, dir)
    # get_LDA_components(k, dir)

    # print("checking stuff aboyut list of features ", len(list_of_features))
    # print("checking stuff aboyut set of features ", set_of_features[1])
    # print("checkning stuff about the matrix ", len(the_matrix[58]))
