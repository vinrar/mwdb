import numpy as np
import os
import json
import glob

import pandas as pd
from utils import timeit

dir = 'data'

fnames = glob.glob("./" + dir + "/*.wrd")
fnames.sort()
for i in range(len(fnames)):
    fnames[i] = os.path.splitext(os.path.basename(fnames[i]))[0]

df = pd.DataFrame(0.0, index=fnames, columns=fnames)


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


@timeit
def calulate_DTW_similarity():
    for i in range(len(fnames)):
        for j in range(i, len(fnames)):
            f1 = json.load(open('./' + dir + '/' + fnames[i] + '.wrd'))
            f2 = json.load(open('./' + dir + '/' + fnames[j] + '.wrd'))
            comp = list(f1.keys())

            temp = []
            for c in comp:
                for senid in f1[c]:
                    w1 = list(np.array(f1[c][str(senid)]['words'])[:, 0])
                    w1_c = list(np.array(f1[c][str(senid)]['words'])[:, 1])
                    w2 = list(np.array(f2[c][str(senid)]['words'])[:, 0])
                    w2_c = list(np.array(f2[c][str(senid)]['words'])[:, 1])
                    dtw_val = dtw(w1, w2, w1_c, w2_c)
                    temp.append(1 / (1 + dtw_val))

            f1 = fnames[i]
            f2 = fnames[j]

            average = np.average(temp)

            df[f1][f2] = average
            df[f2][f1] = average

    df.to_csv('DTW_sim_matrix.csv')


calulate_DTW_similarity()
