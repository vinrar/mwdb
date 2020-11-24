import json
from collections import defaultdict
from random import gauss, uniform

import numpy as np
from numpy.linalg import norm

hash_tables, vectors = [], {}


# generates unit vectors of dimension 'dims' from gaussian distribution
def generate_random_unit_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    # vec = [uniform(0, 1) for i in range(dims)]
    mag = sum(x ** 2 for x in vec) ** .5
    return [x / mag for x in vec]


# returns the single binary code for a gesture and hash unit vector
def get_code_from_vectors(u, v):
    w = np.dot(np.array(u), np.array(v))
    return "1" if w >= 0 else "0"


# returns the cosine similarity score between 2 vectors
def cosine_similarity(u, v):
    return np.dot(u, v) / (norm(u) * norm(v))


def check_nearby_codes(string_1, string_2, n):
    count_diffs = 0
    for a, b in zip(string_1, string_2):
        if a != b:
            count_diffs += 1
            if count_diffs > n:
                return False
    return True


# creates the in-memory index structures using hash_tables and stores the gestures in buckets
def preprocessing(L, k, d):
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
    # print_hash_tables()


def print_hash_tables():
    global hash_tables
    for i, table in enumerate(hash_tables):
        print("\n---------- Hashtable: ", i + 1, "----------")
        for key, val in table.items():
            if key != "unit_vectors":
                print("Code: ", key, ", Bucket: ", val)


# given a query and t, returns the t most similar gestures to the query gesture
def query_algorithm(q, t, k):
    K = k
    global vectors, hash_tables
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
                if check_nearby_codes(key, code, K-k):
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


# given a value of L and k, creates the LSH index structure and handles queries
def locality_sensitive_hashing(L, k):
    dims = len(list(vectors.values())[0])
    preprocessing(L, k, dims)
    query_gesture = input("\n\nEnter query gesture name for similar gestures or q to quit: ")
    while query_gesture != 'Q' and query_gesture != 'q':
        t = int(input("Enter number of similar gestures to be returned, t: "))
        similar_gestures_file = open(query_gesture+'_lsh_similarity.txt', 'w')
        result = query_algorithm(query_gesture, t, k)
        print("\n-----", t, "most similar gestures using Locality Sensitive Hashing Index structure -----\n")
        for i, (key, value) in zip(range(t), result.items()):
            similar_gestures_file.write(key+'\n')
            print("Gesture: ", key, ",\tSimilarity Score: ", value)
        similar_gestures_file.close()
        query_gesture = input("\n\nEnter query gesture name for similar gestures or q to quit: ")


def main():
    global vectors
    L = int(input("Enter the number of layers, L: "))
    k = int(input("Enter the number of hashes per layer, k: "))
    # v_dir = input("Enter directory of vectors: ")
    model = input("Enter vector model - tf or tfidf: ")

    # L, k, v_dir, model = 8, 6, "3_class_gesture_data", "tf"  # default values
    # query_gesture, t = '260', 10

    with open(model + '_vectors.json', 'r') as fp:
        vectors = json.load(fp)

    locality_sensitive_hashing(L, k)


if __name__ == '__main__':
    main()

# Good results for: L = 10 and k = 10
# L, k, v_dir, model = 8, 6, "3_class_gesture_data", "tf"  # default values
# query_gesture, t = '260' and '588', 10
