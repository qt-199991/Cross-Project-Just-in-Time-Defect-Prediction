import scipy.stats as stats
from statistics import mean, stdev
import numpy as np
import pandas as pd
#先repeat表格在进行统计学分析

# def Cohens_d1(data1, data2):
#     m1 = mean(data1)
#     m2 = mean(data2)
#     s1 = stdev(data1)
#     s2 = stdev(data2)
#     d = (m1 - m2) / (np.sqrt((s1 ** 2 + s2 ** 2) / 2))
#     esl = 'None'
#     if np.abs(d) >= 0.4740:
#         esl = 'L' #'Large(L)'
#     elif np.abs(d) < 0.4740 and np.abs(d) >= 0.3300:
#         esl = 'M'  # 'Medium(M)'
#     elif np.abs(d) < 0.3300 and np.abs(d) >= 0.1470:
#         esl = 'S'  # 'Small(S)'
#     elif np.abs(d) < 0.1470:
#         esl = 'N'  # 'Negligible(N)'
#     else:
#         print('d is wrong, please check...')
#
#     return d, esl

def Wilcoxon_signed_rank_test(data1, data2):
    stat, p = stats.wilcoxon(data1, data2)
    return stat, p

def Cohens_d(data1, data2):
    m1 = mean(data1)
    m2 = mean(data2)
    s1 = stdev(data1)
    s2 = stdev(data2)
    d = (m1 - m2) / (np.sqrt((s1 ** 2 + s2 ** 2) / 2))
    esl = 'None'
    if np.abs(d) >= 0.8:
        esl = 'L' #'Large(L)'
        # print('l')
    elif np.abs(d) < 0.8 and np.abs(d) >= 0.5:
        esl = 'M'  # 'Medium(M)'
        # print('m')
    elif np.abs(d) < 0.5 and np.abs(d) >= 0.2:
        esl = 'S'  # 'Small(S)'
        # print('s')
    elif np.abs(d) < 0.2:
        esl = 'N'  # 'Negligible(N)'
        # print('n')
    else:
        print('d is wrong, please check...')

    return d, esl


def WinTieLoss(data1, data2, alpha, r):
    tie = 0
    win = 0
    loss = 0
    for i in range(0, len(data1), r):
        d1 = data1[i: i+r]
        d2 = data2[i: i+r]
        stat, p = Wilcoxon_signed_rank_test(d1, d2)
        d, esl = Cohens_d(d1, d2)
        if p > alpha:
            tie = tie + 1
            print(i/30,'tie')
        elif p <= alpha and d > 0:
            win = win + 1
            print(i/30,'win')
        else:
            loss = loss + 1
            print(i/30,'loss')

    wtl = str(int(win)) + '/' + str(int(tie)) + '/' + str(int(loss))

    return wtl
# data1 = [0.407,0.216,0.243,0.372,0.4,0.311,0.210,0.220,0.226,0.180,0.240,0.206,0.358,0.372,0.341,0.312,0.395,0.389,0.410,0.346,0.365,0.367,0.364,0.333,0.417,0.300,0.206,0.466,0.489,0.266,0.486,0.468,0.378,0.408,0.351,0.475,0.361,0.406,0.393,0.336,0.308,0.584,0.350]
# data2 = [0.324,0.367,0.360,0.286,0.315,0.321,0.219,0.237,0.210,0.215,0.215,0.224,0.362,0.345,0.372,0.329,0.353,0.378,0.411,0.388,0.391,0.403,0.420,0.399,0.490,0.290,0.402,0.407,0.372,0.417,0.536,0.539,0.562,0.534,0.521,0.551,0.412,0.578,0.592,0.475,0.477,0.639,0.396]
for column in ['F1','MCC','G-Mean','Balance','Recall']:
    # path1 = 'F:/朱老师课题/自己的代码/研究点一/svc results/ISKMM.csv'
    path1 = 'E:/360downloads/5000-2000到balance/svc/ISKMM1.csv'
    df = pd.read_csv(path1)
    data1 = []
    data = df[column]
    for i in range(len(data)):
        data1.append(data[i])
    for name in ['BF1','DFAC1','DMDA_JFR1','HISNN1']:
        # path2 = 'F:/朱老师课题/自己的代码/研究点一/svc results/' +name + '.csv'
        path2 = 'E:/360downloads/5000-2000到balance/svc/' + name + '.csv'
        df = pd.read_csv(path2)
        data = df[column]
        data2 = []
        for i in range(len(data)):
            data2.append(data[i])
        # print(column,name,Wilcoxon_signed_rank_test(data1,data2))
        print(column,name,WinTieLoss(data1,data2,0.05,30))   #一般来说大于20是合适做假设检验的