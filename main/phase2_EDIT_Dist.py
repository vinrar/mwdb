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
from collections import defaultdict
import math
import glob
import sys
import copy


def editdist(s, t): # for wrd files
    rows = len(s)+1
    cols = len(t)+1
    
    dist = [[0 for x in range(cols)] for x in range(rows)]

    for row in range(1, rows):
        dist[row][0] = row * 3  

    for col in range(1, cols):
        dist[0][col] = col * 3 
        
    for row in range(1, rows):
        for col in range(1, cols):
            if s[row-1] == t[col-1]:
                cost = 0
            else:
                cost = 0
                for i in range(len(s[row-1])):
                    if s[row-1][i]!=t[col-1][i]:
                        cost+=1

            dist[row][col] = min(dist[row-1][col] + 3, #deletes
                                 dist[row][col-1] + 3, #inserts
                                 dist[row-1][col-1] + cost) # substitution   
    return dist[row][col]

#Compare Gestures using Edit distance for Task 2 User Option 6
dir = 'data'
fnames = glob.glob("./"+dir+"/*.wrd")
fnames.sort()

qfile = '1'

qword = json.load(open('./'+dir+'/'+qfile+'.wrd')) #.wrd
comp = list(qword.keys())

sim = []
for gfile in fnames:
    f = os.path.splitext(os.path.basename(gfile))[0]
    fword = json.load(open('./'+dir+'/'+f+'.wrd')) #.wrd
 
    temp = []
    for c in comp:
        for senid in fword[c]: # for wrd files
            wf = list(np.array(fword[c][str(senid)]['words'])[:,0])
            wq = list(np.array(qword[c][str(senid)]['words'])[:,0])
            temp.append(editdist(wf,wq))

    sim.append((f, 1/(1+np.average(temp))))
  
sim.sort(key = lambda x: x[1], reverse=True) 
print(np.array(sim)[:10])