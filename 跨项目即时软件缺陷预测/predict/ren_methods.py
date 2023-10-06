import math
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
from scipy.stats import *

import matplotlib.pyplot as plt

def arraysort(org_arr):
    length = np.shape(org_arr)[0]
    index = np.zeros(length)
    arr = np.zeros(length)
    for i in range(length):
        index[i] = int(i)
        arr[i] = org_arr[i]
    temp = 0
    thisIndex = 0
    for i in range(length):
        for j in range(length-i-1):
            if(arr[j] < arr[j + 1]):
                temp = arr[j]
                arr[j] = arr[j + 1]
                arr[j + 1] = temp

                thisIndex = index[j]
                index[j] = index[j + 1]
                index[j + 1] = thisIndex
    return index
def getRank(org_arr):
    seq = arraysort(org_arr)
    rank = np.zeros(len(org_arr))
    for j in range(len(org_arr)):
        indexb = int(seq[j])
        rank[indexb] = j+1
    rank2 = getRank2(org_arr)
    return (rank+rank2)/2

def getRank2(org_arr):
    seq = arraysort(reverse(org_arr))
    rank = np.zeros(len(org_arr))
    for j in range(len(org_arr)):
        indexb = int(seq[j])
        rank[indexb] = j+1
    return reverse(rank)

def reverse(org_arr):
    l = len(org_arr)
    reseq = np.zeros(l)
    for j in range(l):
        reseq[j] = org_arr[l-j-1]
    return reseq

def test_friedmanchisquare(data):
    ranks  = []
    for i in range(N):
        ranks.append(getRank(data[i]))
    ranks = np.array(ranks)
    ranks = np.transpose(ranks)
    stat,p = friedmanchisquare(ranks[0], ranks[1], ranks[2], ranks[3], ranks[4])
    print('stat',stat)
    means = []
    for i in range(k):
        means.append(np.average(ranks[i]))
    means = np.array(means)
    means = reverse(means)
    print(means)
    ordinate = np.linspace(5,1,5)

    CD = q_para*math.sqrt(k*(k+1)/(6*N))
    print(CD)

    plt.subplot(1,4)


    plt.hlines(ordinate[0], means[0]-CD/2, means[0]+CD/2, color='#943c39')
    plt.vlines([means[0]-CD/2,means[0]+CD/2], [0,0], [ordinate[0],ordinate[0]], color='#943c39', linewidth=0.5, linestyles='dashed')
    for i in range(1,k):
        plt.hlines(ordinate[i], means[i]-CD/2, means[i]+CD/2, color='#3b6291')
    plt.scatter(means, ordinate, marker='o', color='#3b6291', facecolor='white', alpha=1.0)
    plt.scatter(means[0], ordinate[0], marker='o', color='#943c39', facecolor='white', alpha=1.0)

    plt.yticks( ticks=np.linspace(1,5,5),
        # labels=["DECC","ACkEL","MLWSE","BOOMER","AdaBoost.C2"])
        labels=['1','2','3','4','5'])
    plt.ylim(0,6)
    plt.xticks([])
    # plt.savefig('FTest/'+filenames[test], bbox_inches='tight')
    # plt.clf()
    # plt.show()

# filenames = ["hamming", "coverage", "rankloss", "avgPre"]
N = 30
k = 5
q5_5 = 2.728 #Nemenyi
q5_5_2 = 2.498 #Bonferroni-Dunn
q10_5 = 2.459
q10_5_2 = 2.241
q_para = q5_5_2

plt.figure(figsize=(12, 2), dpi=1000)
a1 = 'E:/360downloads/5000-2000到balance/meanall_measure/nb/F1_nb_.csv'
data = pd.read_csv(a1)
test_friedmanchisquare(data)
data = np.transpose(data)

plt.savefig('ftest', bbox_inches='tight')