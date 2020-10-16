import numpy as np
import os
import json
import glob


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


# v2 = [(7, 0), (9, 0), (2, 0), (9, 0), (2, 0), (7, 0), (6, 0), (5, 0), (4, 0)]
# v1 = [(9, 0), (5, 0), (2, 0), (4, 0), (5, 0), (5, 0), (8, 0)]
# c2 = [7, 9, 2, 9, 2, 7, 6, 5, 4]
# c1 = [9, 5, 2, 4, 5, 5, 8]
# dtw(v1, v2, c1, c2)

dir = 'data'
fnames = glob.glob("./" + dir + "/*.wrd")
fnames.sort()

qfile = '1'  # query file name

qword = json.load(open('./' + dir + '/' + qfile + '.wrd'))  # .wrd
comp = list(qword.keys())

sim = []
for gfile in fnames:
    f = os.path.splitext(os.path.basename(gfile))[0]
    fword = json.load(open('./' + dir + '/' + f + '.wrd'))  # .wrd

    temp = []
    for c in comp:
        for senid in fword[c]:  # for wrd files
            # wrd files
            wf = list(np.array(fword[c][str(senid)]['words'])[:, 0])
            wf_c = list(np.array(fword[c][str(senid)]['words'])[:, 1])
            wq = list(np.array(qword[c][str(senid)]['words'])[:, 0])
            wq_c = list(np.array(qword[c][str(senid)]['words'])[:, 1])
            temp.append(dtw(wf, wq, wf_c, wq_c))

    if np.average(temp) == 0:
        sim.append((f, float('inf')))
    else:
        sim.append((f, 1 / np.average(temp)))
print(sim)

sim.sort(key=lambda x: x[1], reverse=True)
print(np.array(sim)[:10])

# [['1' 'inf']
#  ['11' '0.23549471487099521']
#  ['4' '0.22855609387307063']
#  ['50' '0.21660162989762322']
#  ['22' '0.20961300273316494']
#  ['24' '0.20690511731773592']
#  ['19' '0.19858096123212107']
#  ['48' '0.18673544541987547']
#  ['39' '0.18493266435100839']
#  ['25' '0.1842714730500304']]
