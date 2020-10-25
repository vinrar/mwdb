import os
import pickle
from collections import Counter
import math
import ast
import sys

tf_idf_dict = {}


def get_files_list(dir):
    list_of_files = os.listdir(dir)
    return [ file for file in list_of_files if file.endswith('.wrd')]

def count_of_words_across_dimensions(dict_of_words):
    count_of_words = 0
    for each_dimension in dict_of_words:
        for each_sensor in dict_of_words[each_dimension]:
            count_of_words = count_of_words + len(dict_of_words[each_dimension][each_sensor]["words"])
    return count_of_words

# return the time series data of all 20 sensor data
def read_file_data(dir, list_of_files):
    for each_file in list_of_files:
        file_path = dir + "/" + each_file
        file_handler = open(file_path, 'r')
        content = file_handler.read()
        dict_of_words = ast.literal_eval(content)
        word_count_across_dimensions = count_of_words_across_dimensions(dict_of_words)
        tf_idf_dict[str(each_file.split('.')[0])] = dict_of_words
        get_tf_vector(dict_of_words, word_count_across_dimensions, each_file.split('.')[0], dir)
        # get_tf_idf_vector()
    # print("prinitng the length of tf_idf_dict ", len(tf_idf_dict))

    get_tf_idf_vector(dir)

def get_tf_vector(dict_of_words_across_files, word_count_across_dimensions, file_name, directory):
    the_vectors = {}
    for each_dimension in dict_of_words_across_files:
        for each_sensor in dict_of_words_across_files[each_dimension]:
            list_of_words = []
            for each_word in dict_of_words_across_files[each_dimension][each_sensor]["words"]:
                list_of_words.append(tuple(each_word[0]))
            #print("printing the list of words ", list_of_words)
            set_of_unique_words = list(Counter(tuple(list_of_words)).keys())
            count_of_unique_words = Counter(tuple(list_of_words)).values()
            tf_proportions = [x / (word_count_across_dimensions) for x in count_of_unique_words]
            pairs_words_tf_values = []
            for i in range(len(tf_proportions)):
                pairs_words_tf_values.append((set_of_unique_words[i], tf_proportions[i]))
            if each_dimension in the_vectors:
                the_vectors[each_dimension][each_sensor] = pairs_words_tf_values
            else:
                the_vectors[each_dimension] = {}
                the_vectors[each_dimension][each_sensor] = pairs_words_tf_values

    complete_file_name = "tf_vectors_"+str(file_name)+".txt"
    file_handler = open(directory + "/" + complete_file_name, 'wb')
    pickle.dump(the_vectors, file_handler)


def get_occurence_across_sensors(word, sensor, dimension):
    frequency_of_occurence = 0
    for each_file in tf_idf_dict:
        list_of_words = []
        for each_word in tf_idf_dict[each_file][dimension][sensor]["words"]:
            list_of_words.append(tuple(each_word[0]))
        if word in  list_of_words:
            frequency_of_occurence = frequency_of_occurence + 1
    return frequency_of_occurence

def get_tf_idf_vector(dir):
    N = len(tf_idf_dict)
    for each_file in tf_idf_dict:
        tf_idf_vector = {}
        file_path = dir + "/" + "tf_vectors_"+str(each_file)+".txt"
        file_handler = open(file_path, 'rb')
        tf_vector = pickle.load(file_handler)
        for each_dimension in tf_vector:
            tf_idf_vector[each_dimension] = {}
            for each_sensor in tf_vector[each_dimension]:
                tf_idf_vector[each_dimension][each_sensor] = []
                for each_word in tf_vector[each_dimension][each_sensor]:
                    m = get_occurence_across_sensors(each_word[0], each_sensor, each_dimension)
                    idf = math.log(N/m)
                    # print("prininting the word ", each_word)
                    tf = [every_word[1] for every_word in tf_vector[each_dimension][each_sensor] if every_word[0] == tuple(each_word[0])]
                    # print("prinintg tf vector ", tf_vector[each_dimension][each_sensor])
                    # print("printing the tf value ", tf)
                    tf_idf_vector[each_dimension][each_sensor].append((tuple(each_word[0]), tf[0] * idf))

        complete_file_name = "tfidf_vectors_" + str(each_file) + ".txt"
        file_handler = open(dir + "/" + complete_file_name, 'wb')
        pickle.dump(tf_idf_vector, file_handler)

if __name__ == '__main__':

    if len(sys.argv) < 1:
        print('Run python Task0b.py <Directory> ')
        sys.exit(0)
    directory = sys.argv[1]

    # returns a list of files present in the directory
    list_of_files = get_files_list(directory)

    read_file_data(directory, list_of_files)



