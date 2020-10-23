# -*- coding: utf-8 -*-
"""Phase2_task0b.ipynb

Automatically generated by Colaboratory.
Author: Amey Athale

Original file is located at
    https://colab.research.google.com/drive/1pdYyWBQFOAWsmqBL0-gxjY4CWNT7lxHa
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
import glob
import sys

# Task 4d

#Read sim matrix

df = pd.read_csv('./edit_dist_sim_matrix.csv')

W = np.zeros((df.shape[0],df.shape[0]))

for i in range(df.shape[0]):
    for j in range(1,df.shape[1]):
        if df.iloc[i,j] >= 0:
            W[i,j-1] = df.iloc[i,j]

D = np.diag(np.sum(W, axis=1))
L = D - W

w, v = np.linalg.eig(L)
y = v[:, np.argsort(w)]
p = 10
y = y[:,:p] # This y is the direct input to Kmeans