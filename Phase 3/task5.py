import pandas as pd
import numpy as np


def get_ppr2(k, c, gesture_list, relevant_gestures, irrelevant_gestures, gui=False):
    relevant_gesture_percentage = 1
    non_classified_gesture_percentage = 1 - relevant_gesture_percentage

    non_classified_gestures = list(set(gesture_list) - set(relevant_gestures) - set(irrelevant_gestures))

    # TODO: Make the file name uniform with task 3 input
    df = pd.read_csv("./cosine_sim_matrix.csv")
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
        temp = np.array(df[gesture_list].loc[int(r)])
        topk = temp.argsort()[-k:][::-1]
        for j in topk:
            A.loc[int(r), col_index_map[j]] = temp[j]

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


def get_ppr(qfile, k, m, c, relevant=None):
    df = pd.read_csv("./cosine_sim_matrix.csv")
    df = df.set_index('Unnamed: 0')

    A = df.copy(deep=True)
    for col in A.columns:
        A[col].values[:] = 0

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
    if not relevant:
        vq[A.columns.get_loc(qfile)] = 1
    else:
        length = len(relevant)
        for file in relevant:
            vq[A.columns.get_loc(file)] = 1 / length
    uq = vq
    for i in range(1000):
        uq = (1 - c) * np.matmul(vecA, uq) + c * vq

    print("\nQuery File: ", qfile, ".csv")
    topm = uq.argsort()[-m:][::-1]
    for it in topm:
        print("File: ", A.columns[it], ", PPR: ", uq[it])


if __name__ == '__main__':
    # if len(sys.argv) < 5:
    #     print('Run python task3.py <Directory> <Vector Model> <k> <m> <c>')
    #     sys.exit(0)
    #
    # data_dir = sys.argv[1]
    # k = int(sys.argv[3])
    # m = int(sys.argv[4])
    # c = float(sys.argv[5])
    print("Enter the query gestures")
    qfiles = input()
    print(qfiles)
    k = 30
    m = 10
    c = 0.85
    gesture_list = ['1', '11', '2', '31', '4']
    relevant_gestures = ['1', '4']
    irrelevant_gestures = ['11']
    gesture_not_classified = list(set(gesture_list) - set(relevant_gestures) - set(irrelevant_gestures))
    get_ppr2(qfiles, k, m, c, gesture_list, relevant_gestures, gesture_not_classified,gui=False)

    # print("Enter the relevant gestures (comma separated)")
    # relevant_gestures = input().split(",")
    # print("Enter the irrelavant gestures (comma separated)")
    # irrelevant_gestures = input().split(",")
    # get_ppr(qfiles, k, m, c, relevant)
