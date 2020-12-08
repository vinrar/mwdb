import pickle

with open('non_relevant_current_best_features.pickle', 'rb') as f:
    temp = pickle.load(f)
    # print("prinitng the modified quesry")
    print(len(temp))
    for each_element in temp:
        print(each_element)