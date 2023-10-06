import os
import pandas as pd
import numpy as np

resultspath = 'F:/朱老师课题/自己的代码/研究点一/lr results/'
# resultspath = 'C:/Users/qiutian/Desktop/lr/'
csvfiles = os.listdir(resultspath)
dfiMean = pd.DataFrame()
index = []
for i in range(len(csvfiles)):
    columnname = [csvfiles[i].rstrip(".csv")]
    index.extend(columnname)
    dfi = pd.read_csv(resultspath + csvfiles[i])
    dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
    # print(len(dfi))
    learnername = csvfiles[i].split("_")[-1].rstrip(".csv")
    tempdfiMean = pd.DataFrame(dfi.iloc[0:len(dfi), 3:].mean())  # the first column is "Unnamed: 0"
    # print('tempdfiMean',tempdfiMean)
    dfiMean = pd.concat([dfiMean, tempdfiMean.T])
dfiMean.index = index
# print(dfiMean)
# 完败
# dfiMean.to_csv('F:/朱老师课题/自己的代码/研究点一/deal results/comparemethods/allmethods_nb.csv', index=True)
# 略胜
# dfiMean.to_csv('F:/朱老师课题/自己的代码/研究点一/deal results/comparemethods/allmethods_lr.csv', index=True)
# 性能相对较好
# dfiMean.to_csv('F:/朱老师课题/自己的代码/研究点一/deal results/comparemethods/allmethods_svm.csv', index=True)
# 略胜一点点
# dfiMean.to_csv('F:/朱老师课题/自己的代码/研究点一/deal results/comparemethods/allmethods_lr.csv', index=True)
dfiMean.to_csv('E:/360downloads/5000-2000到balance/compare_allmethods/allmethods_svc.csv', index=True)