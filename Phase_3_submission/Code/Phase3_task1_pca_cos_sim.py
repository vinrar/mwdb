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


f = open('./pca_transformed_tfidf_vectors.json',) 
data = json.load(f) 
print("Generating pca cosine sim matrix")
A = np.zeros((len(data),len(data)))
df = pd.DataFrame(data = A)
df.columns = list(data.keys())
df.index = list(data.keys())
files = list(data.keys())

for i in range(len(files)):
    for j in range(i,len(files)):
        
        # print(i,j)

        if files[i]==files[j]:
            df[files[i]][files[j]] = 1.0
        else:
            cos_sim = 0
            v1 = np.array(data[files[i]])
            v2 = np.array(data[files[j]])
            cos_sim = np.dot(v1,v2) / (np.sum(v1**2)**0.5 * np.sum(v2**2)**0.5)
            df[files[i]][files[j]] = cos_sim
            df[files[j]][files[i]] = cos_sim
print("Saving sim matrix")
df.to_csv("./pca_cosine_sim_tfidf.csv")
print(df.head)