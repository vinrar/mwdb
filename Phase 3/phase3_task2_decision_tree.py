import json
from sklearn.model_selection import train_test_split
import sys

header = []


def unique_values(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts ={}
    for each_row in rows:
        label=each_row[-1]
        if label not in counts:
            counts[label] = 0

        counts[label] +=1

    return counts


def is_value_numeric(value):
    return isinstance(value, int) or isinstance(value, float)


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val =example[self.column]
        if is_value_numeric(val):
            return val>= self.value
        else:
            return val ==self.value

    def __repr__(self):
        condition="=="
        if is_value_numeric(self.value):
            condition =">="
        return "Is %s %s %s?" % (
            header[self.column],condition, str(self.value))


def partition(rows,question):
    true_rows, false_rows = [],[]
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows,false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right,current_uncertainty):
    p = float(len(left)) /(len(left) + len(right))
    return current_uncertainty -p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0])-1

    for col in range(n_features):

        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows,false_rows=partition(rows, question)

            if len(true_rows) ==0 or len(false_rows)== 0:
                continue

            gain =info_gain(true_rows,false_rows,current_uncertainty)

            if gain>= best_gain:
                best_gain,best_question = gain, question

    return best_gain, best_question


class Leaf:

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question=question
        self.true_branch =true_branch
        self.false_branch= false_branch


def build_tree(rows):
    gain, question =find_best_split(rows)

    if gain ==0:
        return Leaf(rows)

    true_rows,false_rows = partition(rows, question)

    true_branch =build_tree(true_rows)

    false_branch= build_tree(false_rows)

    return Decision_Node(question,true_branch, false_branch)


def classify(row,node):
    if isinstance(node,Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_tree(node,spacing=""):
    if isinstance(node,Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + '--> True:')
    print_tree(node.true_branch,spacing + "  ")

    print(spacing + '--> False:')
    print_tree(node.false_branch,spacing + "  ")


def print_leaf(counts):
    total=sum(counts.values()) * 1.0
    probs ={}
    for lbl in counts.keys():
        probs[lbl] =str(int(counts[lbl] / total * 100)) + "%"
    return probs


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('Run python phase3_task2_decision_tree.py <1.PCA 2.SVD>')
        sys.exit(0)

    user_option = sys.argv[1]
    train_data_file_name = ''
    test_data_file_name = ''

    if user_option == '1':
        train_data_file_name = "phase2_task1_transformed_matrix_train_PCA.json"
        test_data_file_name = "phase2_task1_transformed_matrix_test_SVD.json"
    elif user_option == '2':
        train_data_file_name = "phase2_task1_transformed_matrix_train_PCA.json"
        test_data_file_name = "phase2_task1_transformed_matrix_test_SVD.json"

    train_data_file_handler = open(train_data_file_name, 'rb')
    train_data_matrix = json.load(train_data_file_handler)

    test_data_file_handler = open(test_data_file_name, 'rb')
    test_data_matrix = json.load(test_data_file_handler)

    # configure the gesture map here
    # gesture_1 = 'vattene'
    # gesture_2 = 'Combinato'
    # gesture_3 = "D'Accordo"
    gesture_1 = '[1, 31]'
    gesture_2 = '[249, 279]'
    gesture_3 = '[559, 589]'
    range_gesture_map = {}
    range_gesture_map[gesture_1] = "Vattene"
    range_gesture_map[gesture_2] = "Combinato"
    range_gesture_map[gesture_3] = "D'Accordo"
    # 249 - 279
    # 559 - 589
    config_map = {gesture_1: [1, 31], gesture_2: [249, 279], gesture_3: [559, 589]}
    result_map = {}
    for key in config_map:
        config_range = config_map[key]
        for i in range(config_range[0], config_range[1] + 1):
            result_map[str(i)] = key

    train_data = []

    for key in train_data_matrix:
        data_list = train_data_matrix[key]
        data_list.append(result_map[key])
        train_data.append(data_list)

    test_data = []

    for key in test_data_matrix:
        data_list = test_data_matrix[key]
        # data_list.append(result_map[key.split("_")[0]])
        data_list.append(key)
        test_data.append(data_list)

    # training_data, testing_data = train_test_split(train_Data, test_size=0.2)
    training_data, testing_data = train_data, test_data

    key_size = len(training_data[0]) - 1
    # Column labels.
    # These are used only to print the tree.

    for i in range(0, key_size):
        header.append('key_' + str(i))
    # header = ["color", "diameter", "label"]

    header.append("label")

    # # Demo:
    # unique_values(training_data, 0)
    # # Demo:
    # class_counts(training_data)
    #
    # # Now, we'll look at a dataset with many different labels
    # lots_of_mixing = [['1xx'],
    #                   ['6x'],
    #                   ['6xx']]
    # # This will return 0.8
    # gini(lots_of_mixing)
    # #######
    #
    # print(class_counts(training_data))
    #
    # # Demo:
    # # Calculate the uncertainty of our training data.
    # current_uncertainty = gini(training_data)
    # current_uncertainty

    my_tree = build_tree(training_data)
    print_tree(my_tree)

    classify(training_data[0], my_tree)

    # print_leaf(classify(training_data[0], my_tree))
    #
    # print_leaf(classify(training_data[1], my_tree))

    # this is where the last line starts
    total_gestures_in_testing = 0
    correctly_guessed = 0
    output = []
    for row in testing_data:
        total_gestures_in_testing += 1
        actual_label = row[-1]
        classified_label = print_leaf(classify(row, my_tree))

        if result_map[actual_label.split("_")[0]] in classified_label:
            correctly_guessed += 1

        # print("Actual: %s. Predicted Range: %s. Predicted gesture: %s" % (actual_label, list(classified_label.keys())[0], range_gesture_map[list(classified_label.keys())[0]]))
        output.append("Actual: %s. Predicted Range: %s. Predicted gesture: %s" % (
        actual_label, list(classified_label.keys())[0], range_gesture_map[list(classified_label.keys())[0]]))

    output = sorted(output)
    # print(output)
    for each_result in output:
        print(each_result)

    print('Number of gestures in the training data: ' + str(len(training_data)))
    print('Number of gestures in the testing data: ' + str(len(testing_data)))

    print('Accuracy: ' + str(correctly_guessed / total_gestures_in_testing))
