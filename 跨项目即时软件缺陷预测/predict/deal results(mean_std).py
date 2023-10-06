import os
import pandas as pd
import numpy as np
#%%
pwd = os.getcwd()
father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
resultspath = father_path + '/adaboost results/'
print(resultspath)
spath = father_path + '/deal results/Mean/'
if not os.path.exists(spath):
        os.mkdir(spath)

csvfiles = os.listdir(resultspath)
print(csvfiles)
for i in range(len(csvfiles)):
    columnname = [csvfiles[i].rstrip(".csv")]
    dfi = pd.read_csv(resultspath + csvfiles[i])
    dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
    # print(len(dfi))
    learnername = csvfiles[i].split("_")[-1].rstrip(".csv")
    index = []
    for ind in range(0, len(dfi), 30):
        targetindex = dfi["Target"][ind]
        # print('targetindex',targetindex)
        sourceindex = dfi['Source'][ind]
        tempindex = str(sourceindex) + '->' + str(targetindex)
        # print('tempindex',tempindex)
        index.append(tempindex)
    print('index', index)
    dfiMean = pd.DataFrame()
    for m in range(0, len(dfi), 30):
            # print('m',m) #0,30,60
            tempdfiMean = pd.DataFrame(dfi.iloc[m:m+30, 3:].mean() * 100)  # the first column is "Unnamed: 0"
            tempdfistd = pd.DataFrame(dfi.iloc[m:m+30, 3: ].std() * 100)
            tempd = np.vstack((tempdfiMean,tempdfistd))
            tempd = pd.DataFrame(tempd)
            dfiMean = pd.concat([dfiMean, tempd.T])
    dfiMean.index = index
    dfiMean.columns = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision','AUC-std', 'F1-std', 'MCC-std', 'G-Mean-std', 'Balance-std', 'Recall-std', 'PF-std', 'Precision-std']
    # dfiMean.to_csv(spath + "100_Mean_std_" + csvfiles[i],float_format='%.2f')
#对上面的表格进行处理，进行每个参数的对比
# measure = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision']
# csvfile = ['100*Mean_std_AdaBoost.csv','100*Mean_std_AdaBoost.BF.csv','100*Mean_std_AdaBoost.DFAC.csv','100*Mean_std_AdaBoost.HISNN.csv','100*Mean_std_AdaBoost.KWM.csv']
# for i in range(len(measure)):
#     dfiMean = pd.DataFrame()
#     for csv in csvfile:
#         dfi = pd.read_csv('E:/TFboostDF/Mean/' + csv)
#         tempdfiMean = pd.DataFrame(dfi.iloc[:, i+1:i+2])
#         tempdfistd = pd.DataFrame(dfi.iloc[:, i+9:i+10])
#         tempd = np.hstack((tempdfiMean,tempdfistd))
#         tempd = pd.DataFrame(tempd)
#         dfiMean = pd.concat([dfiMean, tempd], axis=1)
#     dfiMean.index = index
#     dfiMean.to_csv('E:/TFboostDF/Mean/' + "Mean_std_" + measure[i] + '.csv',float_format='%.2f')


