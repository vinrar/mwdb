import json
from collections import defaultdict
from random import gauss

import numpy as np
from numpy.linalg import norm
import sys


# hash_tables, vectors = [], {}


def generate_random_unit_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


def get_code_from_vectors(u, v):
    w = np.dot(np.array(u), np.array(v))
    return "1" if w >= 0 else "0"


def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))


def preprocessing(L, k, d, vectors):
    hash_tables = []
    for table in range(L):
        hashtable = defaultdict(list)
        gesture_codes = defaultdict(str)
        for j in range(k):
            unit_v = generate_random_unit_vector(d)
            hashtable["unit_vectors"].append(unit_v)

            for gesture, vector in vectors.items():
                gesture_codes[gesture] += get_code_from_vectors(unit_v, vector)

        for gesture, code in gesture_codes.items():
            hashtable[code].append(gesture)

        hash_tables.append(hashtable)
    return hash_tables
    # print_hash_tables()


def print_hash_tables(hash_tables):
    # global hash_tables
    for i, table in enumerate(hash_tables):
        print("\n---------- Hashtable: ", i + 1, "----------")
        for key, val in table.items():
            if key != "unit_vectors":
                print("Code: ", key, ", Bucket: ", val)


def check_nearby_codes(string_1, string_2, n):
    count_diffs = 0
    for a, b in zip(string_1, string_2):
        if a != b:
            count_diffs += 1
            if count_diffs > n:
                return False
    return True


# given a query and t, returns the t most similar gestures to the query gesture
def query_algorithm(q, vectors, hash_tables, t, k):
    k = int(k)
    K = k

    gestures_to_compare, q_vec = set(), vectors[q]
    print("\nQuery gesture: ", q)

    while len(gestures_to_compare) < t:
        num_buckets = 0
        for hashtable in hash_tables:
            unit_vectors, code = hashtable["unit_vectors"], ""
            for vec in unit_vectors:
                code += get_code_from_vectors(vec, q_vec)
            for key, value in hashtable.items():
                # print("Code: ", code, ", k: ", k, ", code[:k]: ", code[:k])
                # if key.startswith(code[:k]):
                if check_nearby_codes(key, code, K - k):
                    num_buckets += 1
                    for bucket_vector in value:
                        gestures_to_compare.add(bucket_vector)
        k -= 2
        print("\nNumber of buckets searched: ", num_buckets)
        print("Total number of gestures in dataset: ", len(vectors.keys()))
        print("Number of unique gestures compared: ", len(gestures_to_compare))
        # print("List of gestures compared: ", sorted(gestures_to_compare))

        if len(gestures_to_compare) < t:
            print("\nNumber of gestures retrieved are less than ", t, ", searching more buckets...")

    distance_map = {}
    for gesture in gestures_to_compare:
        distance_map[gesture] = cosine_similarity(q_vec, vectors[gesture])

    sorted_x = {k: v for k, v in sorted(distance_map.items(), key=lambda item: item[1], reverse=True)}
    return sorted_x


def locality_sensitive_hashing(L, k, query, t, vectors, hash_tables, gui=False):
    # dims = len(list(vectors.values())[0])
    # preprocessing(L, k, dims)
    result = query_algorithm(query, vectors, hash_tables, t, k)
    print("\n-", t, "most similar gestures using Locality Sensitive Hashing Index structure -\n")
    output = ''
    gesture_list = []
    for i, (k, v) in zip(range(t), result.items()):
        print(i + 1, "Gesture: ", k, ",\tSimilarity Score: ", v)
        output += '{} - Gesture: {},\tSimilarity Score: {}\n'.format(i + 1, k, np.round(v, 3))
        gesture_list.append(str(k))
    if (gui):
        return output, gesture_list


def set_updated_query(rel_gestures, ratio, vectors, query):
    new_vectors = []
    for i, gesture in enumerate(rel_gestures):
        new_vectors.append([element * ratio[i] for element in vectors[gesture]])
    new_vectors = np.array(new_vectors)
    new_vectors = new_vectors.sum(axis=0)
    return list(new_vectors)


def get_appropriate_ratio(gesture_list, rel_gestures, ratio):
    gesture_ratio_map = {}
    for gesture, ratio_val in zip(rel_gestures, ratio):
        gesture_ratio_map[gesture] = ratio_val
    ratio = np.array([gesture_ratio_map[gesture] for gesture in gesture_list])
    ratio = ratio / np.sum(ratio)
    return ratio


def main():
    global vectors
    if len(sys.argv) < 6:
        print('Run python phase3_task3.py <L> <k> <Directory> <Vector Model> <Query Gesture> <t>')
        sys.exit(0)
    L = int(sys.argv[1])
    k = int(sys.argv[2])
    v_dir = sys.argv[3]
    model = sys.argv[4]
    query_gesture = sys.argv[5]
    t = int(sys.argv[6])

    print("L: {}\nk: {}\nDirectory: {}\nVector Model: {}\nQuery Gesture: {}\nt: {}\n".format(L, k, v_dir, model,
                                                                                             query_gesture, t))
    # L, k, v_dir, model = 10, 6, "3_class_gesture_data", "tf"  # default values
    # query_gesture, t = '10', 10
    # query_gesture, t = '260', 10

    with open(model + '_vectors.json', 'r') as fp:
        vectors = json.load(fp)
    dims = len(list(vectors.values())[0])
    hash_tables = preprocessing(L, k, dims, vectors)

    locality_sensitive_hashing(L, k, query_gesture, t, vectors, hash_tables)


if __name__ == '__main__':
    main()

# Good results for:
# L, k, v_dir, model = 8, 6, "3_class_gesture_data", "tf"  # default values
# query_gesture, t = '260' and '588', 10

# =====================================Task 4 =======================================================
