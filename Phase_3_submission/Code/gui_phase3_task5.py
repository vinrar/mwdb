import pandas as pd
import numpy as np


def get_ppr_changing_query(k, m, c, relevant_gestures, gui=False):
    # TODO: Make the file name uniform with task 3 input
    df = pd.read_csv("./pca_cosine_sim_matrix.csv")
    df = df.set_index('Unnamed: 0')

    A = df.copy(deep=True)

    col_index_map = {}
    i = 0
    for col in A.columns:
        A[col].values[:] = 0
        col_index_map[i] = col
        i += 1

    for i in range(df.shape[0]):
        temp = np.array(df.iloc[i, :])
        topk = temp.argsort()[-k:][::-1]
        for j in topk:
            A.iloc[i, j] = df.iloc[i, j]

    for i in range(A.shape[0]):
        A.iloc[i, :] = (A.iloc[i, :] - min(A.iloc[i, :])) / (max(A.iloc[i, :]) - min(A.iloc[i, :]))
        A.iloc[i, :] = A.iloc[i, :] / np.sum(A.iloc[i, :])

    vecA = np.array(A)
    vq = np.zeros(A.shape[0])
    length = len(relevant_gestures)
    for file in relevant_gestures:
        vq[A.columns.get_loc(file)] = (1 / length)
    uq = vq
    for i in range(1000):
        uq = (1 - c) * np.matmul(vecA, uq) + c * vq

    topm = uq.argsort()[-m:][::-1]
    top_result = uq[topm[0]] / 50
    relevant_list = []
    score = []
    for i, it in enumerate(topm):
        if uq[it] >= top_result:
            relevant_list.append(A.columns[it])
            score.append(uq[it])
        print("File: ", A.columns[it], ", PPR: ", uq[it])
    score = np.array(score)
    score = score / np.sum(score)
    return relevant_list, score

    #     output += '{} - Gesture: {},\tSimilarity Score: {}\n'.format(i + 1, A.columns[it], np.round(uq[it], 3))
    # if gui:
    #     return output


def get_ppr2(k, c, gesture_list, relevant_gestures, irrelevant_gestures, gui=False):
    relevant_gesture_percentage = 1
    non_classified_gesture_percentage = 1 - relevant_gesture_percentage

    non_classified_gestures = list(set(gesture_list) - set(relevant_gestures) - set(irrelevant_gestures))

    # TODO: Make the file name uniform with task 3 input
    df = pd.read_csv("./pca_cosine_sim_matrix.csv")
    df = df.set_index('Unnamed: 0')

    A = df.copy(deep=True)
    # Getting the subset from the original similarity matrix
    A = A[gesture_list]
    A = A.loc[A.index.isin(gesture_list)]

    col_index_map = {}
    i = 0
    for col in A.columns:
        A[col].values[:] = 0
        col_index_map[i] = col
        i += 1

    # top k for new similarity graph
    for r in gesture_list:
        temp = np.array(df[gesture_list].loc[r])
        topk = temp.argsort()[-k:][::-1]
        for j in topk:
            A.loc[r, col_index_map[j]] = temp[j]

    for i in range(A.shape[0]):
        A.iloc[i, :] = (A.iloc[i, :] - min(A.iloc[i, :])) / (max(A.iloc[i, :]) - min(A.iloc[i, :]))
        A.iloc[i, :] = A.iloc[i, :] / np.sum(A.iloc[i, :])

    vecA = np.array(A)
    vq = np.zeros(A.shape[0])
    length = len(relevant_gestures)
    for file in relevant_gestures:
        vq[A.columns.get_loc(file)] = (1 / length) * relevant_gesture_percentage
    for file in non_classified_gestures:
        vq[A.columns.get_loc(file)] = (1 / length) * non_classified_gesture_percentage
    uq = vq
    for i in range(1000):
        uq = (1 - c) * np.matmul(vecA, uq) + c * vq

    topm = uq.argsort()[::-1]
    output = ''
    for i, it in enumerate(topm):
        print("File: ", A.columns[it], ", PPR: ", uq[it])
        output += '{} - Gesture: {},\tSimilarity Score: {}\n'.format(i + 1, A.columns[it], np.round(uq[it], 3))
    if gui:
        return output


if __name__ == '__main__':
    # if len(sys.argv) < 5:
    #     print('Run python task3.py <Directory> <Vector Model> <k> <m> <c>')
    #     sys.exit(0)
    #
    # data_dir = sys.argv[1]
    # k = int(sys.argv[3])
    # m = int(sys.argv[4])
    # c = float(sys.argv[5])

    k = 30
    m = 10
    c = 0.85
    relevant_gestures = ['1', '4']
    a = get_ppr_changing_query(k, m, c, relevant_gestures)
    for i in a:
        print(i)

    # print("Enter the relevant gestures (comma separated)")
    # relevant_gestures = input().split(",")
    # print("Enter the irrelavant gestures (comma separated)")
    # irrelevant_gestures = input().split(",")
    # get_ppr(qfiles, k, m, c, relevant)
