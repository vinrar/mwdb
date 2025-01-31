import numpy as np
import json
import pickle


# sample dataset that was manually verified for calculations
# dataset = [[1, 0, 0, 1], [1, 1, 0, 1], [0, 0, 1, 0],[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# map = ["G1", "G2", "G3", "G4", "G5", "G6"]

# dataset = []
# map = []


def read_the_file(the_file):
    dataset = []
    map = []
    with open(the_file) as f:
        data = json.load(f)

    for each_file in data:
        map.append(each_file)
        dataset.append(data[each_file])
    return dataset, map


def convert_to_binary_form(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] != 0:
                dataset[i][j] = 1
    return dataset


def get_ni(dataset):
    list_of_proportions = [0] * len(dataset[0])
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if dataset[i][j] == 1:
                list_of_proportions[j] = list_of_proportions[j] + 1

    return list_of_proportions


def get_initial_similarity(list_of_ni, dataset):
    similarity_scores = [0] * len(dataset)
    length_of_dataset = len(dataset)
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if list_of_ni[j] == length_of_dataset:
                the_value = (length_of_dataset - (list_of_ni[j] - 1)) / list_of_ni[j]
            else:
                the_value = (length_of_dataset - list_of_ni[j]) / list_of_ni[j]
            similarity_scores[i] = similarity_scores[i] + (dataset[i][j] * np.log2(the_value))
    return similarity_scores


def get_improved_params(relevance_results, retrieved_dataset):
    N = len(relevance_results)
    ni_values = [0] * len(retrieved_dataset[0])
    ri_values = [0] * len(retrieved_dataset[0])
    R = relevance_results.count(1)

    for i in range(len(relevance_results)):
        for j in range(len(retrieved_dataset[0])):
            if retrieved_dataset[i][j] == 1:
                ni_values[j] = ni_values[j] + 1

            if (retrieved_dataset[i][j] == 1) and (relevance_results[i] == 1):
                ri_values[j] = ri_values[j] + 1

    pi_values = [0] * len(retrieved_dataset[0])
    ui_values = [0] * len(retrieved_dataset[0])

    for i in range(len(ni_values)):
        pi_values[i] = (ri_values[i] + (ni_values[i] / N)) / (R + 1)
        ui_values[i] = (ni_values[i] - ri_values[i] + (ni_values[i] / N)) / (N - R + 1)

    return pi_values, ui_values


def get_feedback_similarity(pi_values, ui_values, dataset):
    similarity_scores = [0] * len(dataset)
    for i in range(len(dataset)):
        for j in range(len(dataset[0])):
            if ui_values[j] == 0:
                ui_values[j] = 0.0001

            if pi_values[j] == 1:
                pi_values[j] = 0.9999

            if ui_values[j] == 1:
                ui_values[j] = 0.9999

            if pi_values[j] == 0:
                pi_values[j] = 0.0001

            similarity_scores[i] = similarity_scores[i] + (dataset[i][j] * (
                np.log2((pi_values[j] * (1 - ui_values[j])) / (ui_values[j] * (1 - pi_values[j])))))

    return similarity_scores


def get_details_about_query(query, dataset, map):
    index_of_query = map.index(query)
    query_representation = dataset[index_of_query]
    return query_representation


def get_initial_results(list_of_similarity, dataset, map):
    retrieved_dataset = []
    for each_item in list_of_similarity:
        index = map.index(each_item)
        retrieved_dataset.append(dataset[index])
    return retrieved_dataset


def get_feedback_results(list_of_similarity, number_of_required_results, dataset, map):
    numpy_form = np.array(list_of_similarity)
    index_of_largest = np.argsort(numpy_form)[-number_of_required_results:]
    similarity_results = []
    retrieved_dataset = []
    for each_index in index_of_largest:
        similarity_results.append(map[each_index])
        retrieved_dataset.append(dataset[each_index])

    numpy_form = numpy_form[index_of_largest]

    return similarity_results[::-1], retrieved_dataset[::-1], numpy_form[::-1]


def get_input_from_LSH(LSH_ranking_file):
    ranking_list = []
    with open(LSH_ranking_file) as f:
        for line in f:
            ranking_list.append(line.split('\n')[0])
    return ranking_list


# code for making mode 1 work.....the re-ordering of existing results
def get_re_ordered_results(relevance_results, initial_similarity_results, retrieved_dataset):
    pi_values, ui_values = get_improved_params(relevance_results, retrieved_dataset)

    similarity_scores = [0] * len(retrieved_dataset)

    for i in range(len(retrieved_dataset)):
        for j in range(len(retrieved_dataset[0])):
            if ui_values[j] == 0:
                ui_values[j] = 0.0001

            if pi_values[j] == 1:
                pi_values[j] = 0.9999

            if ui_values[j] == 1:
                ui_values[j] = 0.9999

            if pi_values[j] == 0:
                pi_values[j] = 0.0001

            similarity_scores[i] = similarity_scores[i] + (retrieved_dataset[i][j] * (
                np.log2((pi_values[j] * (1 - ui_values[j])) / (ui_values[j] * (1 - pi_values[j])))))

    numpy_form_similarity = np.array(similarity_scores)
    sorted_indexes = numpy_form_similarity.argsort()
    initial_similarity_results = np.array(initial_similarity_results)
    reordered_initial_similarity_results = initial_similarity_results[sorted_indexes]
    numpy_form_similarity = numpy_form_similarity[sorted_indexes]

    return reordered_initial_similarity_results[::-1], numpy_form_similarity[::-1]


def get_modified_query(pi_values, ui_values):
    new_query = []
    for i in range(len(pi_values)):
        if ui_values[i] == 0:
            ui_values[i] = 0.0001

        if pi_values[i] == 1:
            pi_values[i] = 0.9999

        if ui_values[i] == 1:
            ui_values[i] = 0.9999

        if pi_values[i] == 0:
            pi_values[i] = 0.0001

        new_query.append(np.log2((pi_values[i] * (1 - ui_values[i])) / ui_values[i] * (1 - pi_values[i])))

    return new_query


# if __name__ == "__main__":
def generate_output_for_gui(feedback_similarity_results, similarity_scores):
    output = ''
    i = 0
    for gesture, score in zip(feedback_similarity_results, similarity_scores):
        output += '{} - Gesture: {},\tSimilarity Score: {}\n'.format(i + 1, gesture, round(score, 3))
        i += 1
    return output


def get_task4_results(mode, initial_similarity_results, relevance_results, dataset, map, retrieved_dataset,
                      number_of_required_results, feedback_retrieved_dataset, first_feedback):
    # mode 0 for changing the query and giving better results
    # mode 1 for simply re-ordering the results
    # mode = 0

    # this file is generated when phase 3 is run and depending on model(tf or idf) that is taken, we need to read this
    # vector_model = "tf"
    # the_file = "%s_vectors.json" % vector_model

    # this is the initial ranking that is read from LSH..so basically,  LSH takes the query from the user and writes the
    # initial set of results in the ranking file and it is read here...this needs to be read as well
    # LSH_ranking_file = "588_lsh_similarity.txt"

    # reading the dataset and converting things to binary format
    # dataset, map = read_the_file(the_file)
    # dataset = convert_to_binary_form(dataset)

    # first_feedback = True

    # Evaluating similarity based on initial input from LSH
    # initial_similarity_results = get_input_from_LSH(LSH_ranking_file)

    # number_of_required_results = len(initial_similarity_results)
    # retrieved_dataset = get_initial_results(initial_similarity_results, dataset, map)

    # print("Initial similarity results are")
    # print(initial_similarity_results)

    # mode 0 denoting the workflow where the query is modified based on the feedback and
    # we get progressively better results
    if mode == 0:
        # print("Enter the relevance results")
        # relevance_results = [int(x) for x in input().split()]
        # relevance_results = [0, 1, 0, 0, 0, 1, 0, 1, 1, 0]
        if first_feedback:
            pi_values, ui_values = get_improved_params(relevance_results, retrieved_dataset)
        else:
            pi_values, ui_values = get_improved_params(relevance_results, feedback_retrieved_dataset)

        feedback_similarity = get_feedback_similarity(pi_values, ui_values, dataset)
        feedback_similarity_results, feedback_retrieved_dataset, similarity_scores = get_feedback_results(
            feedback_similarity, number_of_required_results, dataset, map)
        # print("prinitng the feedback results")
        # print(feedback_similarity_results)
        # print("prinitngg the similarity scores")
        # print(similarity_scores)
        output = generate_output_for_gui(feedback_similarity_results, similarity_scores)
        return output, feedback_retrieved_dataset, feedback_similarity_results
    # mode 1 denoting the workflow where the results are re-ordered
    else:
        # print("running the re-ordering of results workflow")
        # print("Enter the relevance results")
        # relevance_results = [int(x) for x in input().split()]
        re_ordered_results, reordered_similarity_scores = get_re_ordered_results(relevance_results,
                                                                                 initial_similarity_results,
                                                                                 retrieved_dataset)
        # print("printing re-ordered results")
        # print(re_ordered_results)
        # print(reordered_similarity_scores)
        output = generate_output_for_gui(re_ordered_results, reordered_similarity_scores)
        return output, None, re_ordered_results
