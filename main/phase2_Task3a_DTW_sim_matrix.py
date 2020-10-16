import numpy as np
import os
import json
import glob

dir = 'data'
comp = ['X', 'Y', 'Z', 'W']
fnames = glob.glob("./" + dir + "/*.wrd")
fnames.sort()
sim_mat = np.zeros((len(fnames), len(fnames)))  # sim matrix


def dtw(vector1, vector2, cost1, cost2):
    assert len(vector1) == len(cost1)
    assert len(vector2) == len(cost2)
    row = len(vector1) + 1
    col = len(vector2) + 1
    cost1.reverse()
    dp = [[float('inf')] * col for i in range(row)]
    dp[row - 1][0] = 0
    for i in range(row - 2, -1, -1):
        for j in range(1, col):
            cost = 0
            if vector1[i] != vector2[j - 1]:
                cost = abs(cost1[i] - cost2[j - 1])
            dp[i][j] = cost + min(dp[i + 1][j], dp[i][j - 1], dp[i + 1][j - 1])

    return dp[0][-1]


for i in range(len(fnames)):
    for j in range(i, len(fnames)):
        f1 = json.load(open(fnames[i]))
        f2 = json.load(open(fnames[j]))

        temp = []
        for c in comp:
            for senid in f1[c]:
                w1 = list(np.array(f1[c][str(senid)]['words'])[:, 0])
                w1_c = list(np.array(f1[c][str(senid)]['words'])[:, 1])
                w2 = list(np.array(f2[c][str(senid)]['words'])[:, 0])
                w2_c = list(np.array(f2[c][str(senid)]['words'])[:, 1])
                temp.append(dtw(w1, w2, w1_c, w2_c))

        f1 = os.path.splitext(os.path.basename(fnames[i]))[0]
        f2 = os.path.splitext(os.path.basename(fnames[j]))[0]

        if np.average(temp) == 0:
            sim_mat[int(f1) - 1, int(f2) - 1] = float('inf')
        else:
            avg = 1 / np.average(temp)
            sim_mat[int(f1) - 1, int(f2) - 1] = avg
            sim_mat[int(f2) - 1, int(f1) - 1] = avg

np.savetxt('task3a_DTW_sim_matrix.txt', sim_mat)
