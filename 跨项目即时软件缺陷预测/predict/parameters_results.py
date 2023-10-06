# import os
# import pandas as pd
# import numpy as np
# #%%
# pwd = os.getcwd()
# father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
# print(father_path)
# resultspath = 'E:/360downloads/parameters/'
# # resultspath = father_path + '/deal results/parameters_affect/'
# # print(resultspath)
#
# csvfiles = os.listdir(resultspath)
# # print(csvfiles)
# index = []
# dfiMean = pd.DataFrame()
# for i in range(len(csvfiles)):
#     columnname = [csvfiles[i].rstrip(".csv")]
#     dfi = pd.read_csv(resultspath + csvfiles[i])
#     dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
#     learnername = csvfiles[i].split("_")[-3].rstrip(".csv") + '_' + csvfiles[i].split("_")[-1].rstrip(".csv")
#     index.append(learnername)
#     tempdfiMean = pd.DataFrame(dfi.iloc[:, 5:].mean())  # the first column is "Unnamed: 0"
#     dfiMean = pd.concat([dfiMean, tempdfiMean.T])
# dfiMean.index = index
# dfiMean.columns = ['KMM', 'BSW', 'AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision']
# dfiMean.to_csv('E:/360downloads/parameters/parameters_results.csv')

# import os
# import pandas as pd
# import numpy as np
# pwd = os.getcwd()
# # father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
# # print(father_path)
# resultspath = 'E:/360downloads/parameters/'
# # resultspath = father_path + '/deal results/parameters_affect/'
# # print(resultspath)
#
# csvfiles = os.listdir(resultspath)
# # print(csvfiles)
# index = []
# dfiMean = pd.DataFrame()
# for i in range(len(csvfiles)):
#     columnname = [csvfiles[i].rstrip(".csv")]
#     dfi = pd.read_csv(resultspath + csvfiles[i])
#     dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
#     kww = csvfiles[i].split("_")[-3].rstrip(".csv")
#     bsw = csvfiles[i].split("_")[-1].rstrip(".csv")
#     # index.append(kww)
#     tempdfiMean = pd.DataFrame(dfi.iloc[:, 5:].mean())  # the first column is "Unnamed: 0"
#     tempdfiMean = tempdfiMean.values.tolist()
#     kwm1 = []
#     kwm1.append(float(kww))
#     bsw1 = []
#     bsw1.append(float(bsw))
#     tempdfiMean.append(kwm1)
#     tempdfiMean.append(bsw1)
#     tempdfiMean = pd.DataFrame(tempdfiMean)
#     dfiMean = pd.concat([dfiMean, tempdfiMean.T])
# # dfiMean.index = index
# dfiMean.columns = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision', 'KMM', 'BSW']
# dfiMean.to_csv('E:/360downloads/parameters/parameters_results.csv')

import os
import pandas as pd
import numpy as np

# resultspath = 'E:/360downloads/kwm-parameters/'
# resultspath = 'E:/360downloads/svc-nosmote/'
resultspath = 'E:/360downloads/svc-parameters/kwm/'
csvfiles = os.listdir(resultspath)
# print(csvfiles)
index = []
dfiMean = pd.DataFrame()
for i in range(len(csvfiles)):
    columnname = [csvfiles[i].rstrip(".csv")]
    dfi = pd.read_csv(resultspath + csvfiles[i])
    dfi = dfi.replace(np.nan, 0)  # replace the NAN to 0
    kww = csvfiles[i].split("_")[-3].rstrip(".csv")
    bsw = csvfiles[i].split("_")[-1].rstrip(".csv")
    # index.append(kww)
    tempdfiMean = pd.DataFrame(dfi.iloc[:, 5:].mean())  # the first column is "Unnamed: 0"
    tempdfiMean = tempdfiMean.values.tolist()
    kwm1 = []
    kwm1.append(float(kww))
    bsw1 = []
    bsw1.append(float(bsw))
    tempdfiMean.append(kwm1)
    tempdfiMean.append(bsw1)
    tempdfiMean = pd.DataFrame(tempdfiMean)
    dfiMean = pd.concat([dfiMean, tempdfiMean.T])
# dfiMean.index = index
dfiMean.columns = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision', 'KMM', 'BSW']
# dfiMean.to_csv('E:/360downloads/kwm-parameters/parameters-results.csv')
# dfiMean.to_csv('E:/360downloads/svc-nosmote/parameters-results.csv')
dfiMean.to_csv('E:/360downloads/svc-parameters/kwm/parameters-results.csv')