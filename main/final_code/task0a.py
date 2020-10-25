"""
Multimedia Web Databases - Fall 2020: Project Group 1
Author: Amey Athale
This is the program for task 0a of Phase 2 of the project
"""

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
import math
import glob
import sys
from scipy.integrate import quad
import copy


dir = 'data'
wfnames = glob.glob("./"+dir+"/W/*.csv")
xfnames = glob.glob("./"+dir+"/X/*.csv")
yfnames = glob.glob("./"+dir+"/Y/*.csv")
zfnames = glob.glob("./"+dir+"/Z/*.csv")

wfnames.sort()
xfnames.sort()
yfnames.sort()
zfnames.sort()

comp = ['W','X','Y','Z']

def gauss(x,mu,sig):
    denom = (1/sig*np.sqrt(2*np.pi))
    num = np.power(np.e, (-0.5*((x-mu)/sig)**2))
    return num/denom

w = 3
s = 2
r = 3
mu = 0
sig = 0.25
A = quad(gauss,-1,1,args=(mu,sig))
lens=[]
for j in range(1,2*r+1):
    ll=(j-r-1)/r
    ul=(j-r)/r
    interg = quad(gauss,ll,ul,args=(mu,sig))
    lens.append(2 * (interg[0]/A[0]))

newr=[]
newr.append(-1)
for j in range(len(lens)):
    newr.append(newr[j]+lens[j])

ranges=[]
for j in range(len(newr)-1):
    if j==r:
        ranges.append((0,newr[j+1]))
    elif j==r-1:
        ranges.append((newr[j],0))
    elif j==len(newr)-2:
        ranges.append((newr[j],1))
    else:
        ranges.append((newr[j],newr[j+1]))
# print(ranges)

for gfile in wfnames: #file
    f = os.path.splitext(os.path.basename(gfile))[0]
    print(f)
    wrd_dict = {}
    for c in comp: #components
        df = pd.read_csv("./"+dir+"/"+c+"/"+f+".csv", header=None)
        avg = 0
        stdev = 0
        comp_dict = {}
        for i in range(df.shape[0]): #sensors
            sensor_dict = {}
            sensor_dict['avg'] = np.average(df.iloc[i,:])
            sensor_dict['stdev'] = np.std(df.iloc[i,:])

            df.iloc[i,:] = 2 * ((df.iloc[i,:] - min(df.iloc[i,:])) / (max(df.iloc[i,:]) - min(df.iloc[i,:]))) - 1
            # quant = copy.deepcopy(df)
            quant = pd.DataFrame(index=range(df.shape[0]),columns=range(df.shape[1]))
            for k in range(df.shape[1]):
                for j in range(len(ranges)):
                    if df.iloc[i,k] >= ranges[j][0] and df.iloc[i,k] < ranges[j][1]:
                        # quant.iloc[i,k] = j
                        # quant.iloc[i,k] = [j, (ranges[j][0]+ranges[j][1])/2]
                        quant.iloc[i,k] = [j+1, (ranges[j][0]+ranges[j][1])/2]
                        break
                    if df.iloc[i,k] == 1:
                        # quant.iloc[i,k] = len(ranges) - 1
                        # quant.iloc[i,k] = [len(ranges) - 1, (ranges[-1][0] + ranges[-1][1]) / 2]
                        quant.iloc[i,k] = [len(ranges), (ranges[-1][0] + ranges[-1][1]) / 2]
                        break
            
            words = []
            for j in range(0,df.shape[1],s):
                if (j+w-1 <= df.shape[1]-1):
                    # winQ = [quant.iloc[i,k] for k in range(j,j+w)]
                    winSym = [quant.iloc[i,k][0] for k in range(j,j+w)] #H
                    # winN = [df.iloc[i,k] for k in range(j,j+w)]
                    winQuant = [quant.iloc[i,k][1] for k in range(j,j+w)]
                    avgQuant = np.average(np.array(winQuant)) #G
                    # avgSym = 0
                    # for k in range(len(ranges)):
                    #     if avgN >= ranges[k][0] and avgN < ranges[k][1]:
                    #         avgSym = k
                    #         break
                    #     if avgN == 1:
                    #         avgSym = len(ranges) - 1
                    #         break
                    # words.append([winQ,avgN,avgSym])
                    words.append([winSym,avgQuant])
            sensor_dict['words'] = words 

            comp_dict[i] = sensor_dict
        wrd_dict[c] = comp_dict

    json.dump(wrd_dict, open('./'+dir+'/'+f+'.wrd','w'))