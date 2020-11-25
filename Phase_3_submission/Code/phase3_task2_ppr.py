import math
import os
import pickle
from collections import Counter
from collections import defaultdict
import sys
import glob
import json
import pandas as pd
import numpy as np
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

def get_ppr_predictions(data_dir, k, m, c):
    df = pd.read_csv("./pca_cosine_sim_matrix.csv")
    # df = pd.read_csv("./cosine_sim_matrix.csv")
    df = df.set_index('Unnamed: 0')

    A = df.copy(deep=True)
    for col in A.columns:
        A[col].values[:] = 0

    for i in range(df.shape[0]):
        temp = np.array(df.iloc[i,:])
        topk = temp.argsort()[-k:][::-1]
        for j in topk:
            A.iloc[i,j] = df.iloc[i,j]

    for i in range(A.shape[0]):
        A.iloc[i,:] = (A.iloc[i,:] - min(A.iloc[i,:])) / (max(A.iloc[i,:]) - min(A.iloc[i,:]))
        A.iloc[i,:] = A.iloc[i,:] / np.sum(A.iloc[i,:])

    df_labels = pd.read_excel("./"+data_dir+"/labels.xlsx", sheet_name=None, header=None)
    df_labels['Sheet1'].columns = ['name', 'class']
    df_labels['Sheet1']['name'] = df_labels['Sheet1'].name.astype(str)
    label_dict = dict(zip(df_labels['Sheet1'].iloc[:,0], df_labels['Sheet1'].iloc[:,1]))
    ground_truth = {}

    files = np.array(df_labels['Sheet1'].iloc[:,:])
    add_files = list(A.columns)

    vecA = np.array(A)
    pred = {}


    # for i in range(files.shape[0]):
    #     qfile = str(files[i,0])
    for i in range(len(add_files)):
        qfile = str(add_files[i])

        vq = np.zeros(A.shape[0])
        vq[A.columns.get_loc(qfile)] = 1
        uq = vq
        uqlist = []
        uqlist.append(uq)

        for j in range(5):
            uq = (1 - c) * np.matmul(vecA,uq) + c * vq
            uqlist.append(uq)
        
        # topm = uq.argsort()[-m:][::-1]
        topm = uq.argsort()[::-1]
        temp = []
        count = 0
        for j in range(len(topm)):
            if str(A.columns[topm[j]]) in label_dict.keys() and count<m:
                temp.append(label_dict[str(A.columns[topm[j]])])
                count += 1

        values, count = np.unique(temp, return_counts=True)
        pred[qfile] = values[np.argmax(count)]

        if qfile not in label_dict:
            ground_truth[qfile] = label_dict[qfile.split('_')[0]]
        else:
            ground_truth[qfile] = label_dict[qfile]

    for it in pred:
        print("File: ",it,", Actual: ",ground_truth[it],", Predicted: ",pred[it])

    count = 0
    for it in pred:
        if pred[it] == ground_truth[it]:
            count += 1
    print("Accuracy = ",count/len(ground_truth))

    # for it in pred:
    #     print("File: ",it,", Actual: ",label_dict[it],", Predicted: ",pred[it])

    # count = 0
    # for it in pred:
    #     if pred[it] == label_dict[it]:
    #         count += 1
    # print("Accuracy = ",count/len(label_dict))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Run python phase3_task2_ppr.py <Directory> <k> <m> <c>')
        sys.exit(0)

    data_dir = sys.argv[1]
    k = int(sys.argv[2])
    m = int(sys.argv[3])
    c = float(sys.argv[4])

    print("Directory: {}\nk: {}\nm: {}\nc: {}\n".format(data_dir, k, m, c))
    
    get_ppr_predictions(data_dir, k, m, c)