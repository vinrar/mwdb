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

def get_ppr(data_dir, qfiles, k, m, c):
    df = pd.read_csv("./pca_cosine_sim_matrix.csv")
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

    vecA = np.array(A)

    for qfile in qfiles:
        qfile = str(qfile)
        vq = np.zeros(A.shape[0])
        vq[A.columns.get_loc(qfile)] = 1
        uq = vq
        uqlist = []
        uqlist.append(uq)

        for i in range(5):
            uq = (1 - c) * np.matmul(vecA,uq) + c * vq
            uqlist.append(uq)

        diff = []
        for i in range(len(uqlist)-1):
            diff.append(np.linalg.norm(uqlist[i+1] - uqlist[i]))

        print("Plotting convergence")
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        ax = plt.axes()
        x = np.arange(1,6)
        ax.plot(x, diff)
        plt.title("Convergence")
        plt.xlabel("Number of Iterations")
        plt.ylabel("Change in uq")
        plt.savefig("./"+qfile+"csv_converge"+".pdf")
        # plt.show()
        plt.close()

        print("\nQuery File: ",qfile,".csv")
        topm = uq.argsort()[-m:][::-1]
        for it in topm:
            print("File: ",A.columns[it], ", PPR: ",uq[it])

        comp = ['W','X','Y','Z']
        from matplotlib import colors
        plot_colors = dict(colors.CSS4_COLORS)
        temp = np.array(list(plot_colors.keys()))
        np.random.shuffle(temp)

        for gest in topm:
            f = str(A.columns[gest])
            print("Plotting ",qfile+"csv_"+f)

            fig, ax = plt.subplots(2,2)
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            fig.suptitle(f)
            fig.text(0.5, 0.04, "timeseries", ha='center', va='center')
            fig.text(0.06, 0.5, "value", ha='center', va='center', rotation='vertical')

            axl = [(0,0), (0,1), (1,0), (1,1)]

            for it in range(len(comp)):
                df = pd.read_csv("./"+data_dir+"/"+comp[it]+"/"+f+".csv", header=None)
                for j in range(20):
                    df = df.rename(columns={j:"Series"+str(j)})

                x = np.arange(1, df.shape[1]+1)

                for i in range(df.shape[0]):
                    l = "Series "+str(i+1)
                    y = df.iloc[i,:]
                    temp_c = np.random.choice(list(plot_colors.keys()), replace=False)
                    ax[axl[it]].plot(x, y, label=l, color=temp[i])
                
                ax[axl[it]].set_title(comp[it])

            legend_val = ax[0,1].legend(title = "Legend", bbox_to_anchor = (1.05,1), loc='upper left', ncol=2)

            filename = "./"+qfile+"csv_"+f+".pdf"
            plt.savefig(filename, bbox_inches='tight', format='pdf', dpi=1000)

            # plt.show()
            plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Run python phase3_task1.py <Directory> <k> <m> <c>')
        sys.exit(0)

    data_dir = sys.argv[1]
    k = int(sys.argv[2])
    m = int(sys.argv[3])
    c = float(sys.argv[4])

    print("Directory: {}\nk: {}\nm: {}\nc: {}\n".format(data_dir, k, m, c))

    print("Enter the query gestures without extension (space separated):")
    qfiles = [str(x) for x in input().split()]  
    print(qfiles)
    
    get_ppr(data_dir, qfiles, k, m, c)