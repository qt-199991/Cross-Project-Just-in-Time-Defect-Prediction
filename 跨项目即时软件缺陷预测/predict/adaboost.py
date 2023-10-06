import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    VotingClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter

from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import warnings

import tca

warnings.filterwarnings("ignore")

from tca import TCA

from AlgoCPCD import TSboostDF


scores = pd.DataFrame(columns=['Source', 'Target', 'AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision'])


# scores = pd.DataFrame(columns=['NB', 'TSboostDF', 'TNB', 'adaBoost', 'adaBoost.NC', 'NN'])
def GMean_and_Balance(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    auc = roc_auc_score(y_true,y_pred)

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    pf = FP / (FP + TN)

    GMean = np.sqrt(recall * (1 - pf))

    Balance = 1 - (np.sqrt(pf ** 2 + (1 - recall) ** 2) / np.sqrt(2))

    return auc, Balance, GMean, recall, pf, precision

dataset = np.asarray(['gerrit','go','jdt', 'openstack','platform','qt'])
def datasetMaker(dataset_name):
    path = 'E:/JIT/Data_Extraction/git_base/datasets/' + dataset_name + '/cross/' + dataset_name + '_k_feature.csv'
    df = pd.read_csv(path)
    col_name = df.columns.values
    X = df[col_name[5:]].values
    X = np.array(X)
    X = X[-5000:]
    y = df[col_name[3]].values
    y = np.array(y)
    y = y[-5000:]
    y[y > 0] = 1

    return X, y
last = 0

for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    for j in range(len(dataset)):  # 遍历数据库，选定源项目
        if i == j: continue
        for r in range(30):
            arr = []
            arr.append(dataset[j])
            arr.append(dataset[i])
            X_sample, y_sample = datasetMaker(dataset[j])
            # print('X_sample',X_sample.shape)
            X_target, y_target = datasetMaker(dataset[i])
            # print('X_target',X_target.shape)

            #
            # X, y = X_sample, y_sample
            #
            # X_sample, X_target = tca.TCA().fit(X_sample, X_target)  # 先进行TCA转换
            #
            # sc = StandardScaler()
            #
            # X_sample = sc.fit_transform(X_sample)
            # X_target = sc.transform(X_target)

            # imb = ADASYN()
            imb = SMOTE()
            X_sample, y_sample = imb.fit_resample(X_sample, y_sample)  #对样本进行过采样处理
            # print('X_sample',X_sample.shape)
            # print('y_sample',Counter(y_sample))
            clf = AdaBoostClassifier(base_estimator = GaussianNB(),n_estimators = 20,learning_rate=0.1)
            # clf = AdaBoostClassifier()
            clf.fit(X_sample,y_sample)

            y_pred = clf.predict(X_target)
            # print(y_pred)
            auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_pred)
            arr.append(auc)
            arr.append(f1_score(y_target, y_pred))
            arr.append(matthews_corrcoef(y_target, y_pred))
            arr.append(GMean)
            arr.append(Balance)
            arr.append(recall)
            arr.append(pf)
            arr.append(precision)

            # print(arr)

            scores.loc[last] = arr
            last = last + 1

# scores.to_csv('Results/allmethods_AdaBoost_2.csv') #去掉smote，结果很差，且对于同一组实验结果是一样的
scores.to_csv('Results/allmethods_AdaBoost_3.csv') #加上所有预处理步骤




