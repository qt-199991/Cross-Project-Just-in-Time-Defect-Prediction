import numpy as np
# import math
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from sklearn.utils import shuffle
from collections import Counter
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import os
from pofb20 import get_pofb20, get_efforts
from sklearn import linear_model,ensemble
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt','gerrit','matplotlib','brackets','camel'])
# dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt'])
# scores = pd.DataFrame(columns=['Source', 'Target', 'AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision'])
scores = pd.DataFrame(columns=['Source','Target', 'pofb20'])
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

def datasetMaker(dataset_name):
    path = '/home/cloudam/qt/data/' + dataset_name + '.csv'
    df = pd.read_csv(path)
    col_name = df.columns.values
    X = df[col_name[5:]].values
    X = np.array(X)
    X = X[-5000:]
    y = df[col_name[3]].values
    y = np.array(y)
    y = y.astype(int)
    y = y[-5000:]
    y[y > 0] = 1

    return X, y
def trainMaker(dataset_name, r):
    path = '/home/cloudam/qt/data/' + dataset_name + '.csv'
    df = pd.read_csv(path)
    df = df[-5000:]
    # print(df)
    gbr = df.groupby('bug', group_keys=0)
    X_pos = pd.DataFrame()
    X_neg = pd.DataFrame()
    # X_pos_index = []
    # X_neg_index = []
    for name, group in gbr:
        # print(len(group))
        typicalFracDict = {1: 1, 0: 1}
        frac = typicalFracDict[name]  # 0.9
        # random sample according to the row
        result = group.sample(n=None, frac=frac, replace=False, weights=None, random_state= r + 1,
                              axis=0)  # 不可放回的抽样，每类别抽取90%的数据
        if name == 1:
            # frac_bug = len(result) / len(group)   # bug_num/ bug_num_ori≈frac
            X_pos = result
            X_pos_index = X_pos.index.values
            X_pos_index.sort()  # ranking from small to large
            # print(X_pos)
        else:
            X_neg = result
            X_neg_index = X_neg.index.values
            X_neg_index.sort()
            # print(X_neg)
    col_name = X_pos.columns.values
    X_pos_train = X_pos[col_name[5:]].values
    X_pos_train = np.array(X_pos_train)
    y_pos_train = X_pos[col_name[3]].values
    y_pos_train = np.array(y_pos_train)
    y_pos_train[y_pos_train > 0] = 1
    X_neg_train = X_neg[col_name[5:]].values
    X_neg_train = np.array(X_neg_train)
    y_neg_train = X_neg[col_name[3]].values
    y_neg_train = np.array(y_neg_train)
    y_neg_train[y_neg_train > 0] = 1
    Sample_tr = np.concatenate((X_pos_train, X_neg_train), axis=0)
    Sample_yr = np.concatenate((y_pos_train, y_neg_train), axis=0)
    X_train,y_train = shuffle(Sample_tr,Sample_yr) #这样就返回了最后的训练集
    return X_train, y_train

def BF(X_tr, y_tr, X_te, y_te, k):

    dataSetSize = X_tr.shape[0]
    dataSetwide = X_tr.shape[1]
    testSetSize = len(y_te)

    # 计算目标项目中每个的实例的最近邻点
    selectinstances = np.zeros(shape=(len(X_te)*k, dataSetwide))
    selectlabels = np.zeros(len(X_te)*k)  # ValueError: could not broadcast input array from shape (3) into shape (3,1)
    # print('selectlabels', selectlabels.shape)
    i = 0
    for inX in X_te:
        i += 1
        diffMat = np.tile(inX, (dataSetSize, 1)) - X_tr  # 在行上重复dataSetSize次，列上重复1次
        sqDiffMat = diffMat ** 2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances ** 0.5  # 计算欧式距离  sqrt(Σ(x-y)^2),一个测试实例对应的距离有dataSetSize个

        # distances_te.append(distances)  # 所有的欧氏距离列表
        sortedDistIndicies = distances.argsort().tolist()  # 数组升序排序返回索引，并数组转换为列表
        selectlist = sortedDistIndicies[: k]  # 选择距离最近的k个值的索引列表
        X_tr_select = X_tr[selectlist]
        # deletlist = sortedDistIndicies[k: ]  # 删除距离远的值的索引列表
        # X_tr_select = np.delete(X_tr, deletlist, axis=0) # 与上两行结果等价
        y_tr_select = y_tr[selectlist]
        selectinstances[(i-1)*k: i*k] = X_tr_select
        selectlabels[(i-1)*k: i*k] = y_tr_select

    return selectinstances, selectlabels
last = 0
for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    for j in range(len(dataset)):  # 遍历数据库，选定源项目
        for r in range(2):
            if i == j: continue
            arr = []
            arr.append(dataset[j])
            arr.append(dataset[i])
            efforts = get_efforts(dataset[i])
            X_sample, y_sample = trainMaker(dataset[j], r)
            X_target, y_target = datasetMaker(dataset[i])
            # print('y_sample', Counter(y_sample))
            # print('y_target', Counter(y_target))

            sc = StandardScaler()
            X_sample = sc.fit_transform(X_sample)
            X_target = sc.transform(X_target)

            X_train_new, y_train_new = BF(X_sample, y_sample, X_target, y_target, k=10)
            # print('y_train_new', Counter(y_train_new))
            imb = SMOTE()
            X_train_new, y_train_new = imb.fit_resample(X_train_new, y_train_new)
            # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
            # clf = GaussianNB(priors=None)
            # clf = svm.SVC()
            # clf = svm.SVC(class_weight = 'balanced')
            # clf = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')
            clf = ensemble.RandomForestClassifier(n_estimators=100)
            clf.fit(X_train_new,y_train_new)
            y_pred = clf.predict(X_target)
            pofb20 = get_pofb20(np.array(y_target), y_pred, efforts)
            # print(y_pred)
            arr.append(round(pofb20 * 100, 2))
            # auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_pred)
            # arr.append(round(auc * 100, 2))
            # arr.append(round(f1_score(y_target, y_pred) * 100, 2))
            # arr.append(round(matthews_corrcoef(y_target, y_pred) * 100, 2))
            # arr.append(round(GMean * 100, 2))
            # arr.append(round(Balance * 100 , 2))
            # arr.append(round(recall * 100, 2))
            # arr.append(round(pf * 100, 2))
            # arr.append(round(precision * 100, 2))
            print(arr)


            scores.loc[last] = arr
            last = last + 1
scores.to_csv('/home/cloudam/qt/pofb20/BF_pofb20_rf.csv')

# last = 0
# j = 0  #训练
# i = 3  #测试
# for r in range(20,50):
#     arr = []
#     arr.append(dataset[j])
#     arr.append(dataset[i])
#     X_sample, y_sample = trainMaker(dataset[j], r)
#     X_target, y_target = datasetMaker(dataset[i])
#     print('y_sample', Counter(y_sample))
#     print('y_target', Counter(y_target))

#     sc = StandardScaler()
#     X_sample = sc.fit_transform(X_sample)
#     X_target = sc.transform(X_target)

#     # imb = SMOTE()
#     # X_sample, y_sample = imb.fit_resample(X_sample, y_sample)  # 对源项目进行过采样处理
#     X_train_new, y_train_new = BF(X_sample, y_sample, X_target, y_target, k=10)
#     print('y_train_new', Counter(y_train_new))

#     # sc = StandardScaler()
#     # X_train_new = sc.fit_transform(X_train_new)
#     # X_target = sc.transform(X_target)

#     # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
#     # clf = GaussianNB(priors=None)
#     # clf = svm.SVC()
#     # clf = svm.SVC(class_weight = 'balanced')
#     clf = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')
#     # clf = ensemble.RandomForestClassifier(random_state= r + 1, n_estimators=100)
#     clf.fit(X_train_new,y_train_new)
#     y_pred = clf.predict(X_target)
#     auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_pred)
#     arr.append(auc)
#     arr.append(f1_score(y_target, y_pred))
#     arr.append(matthews_corrcoef(y_target, y_pred))
#     arr.append(GMean)
#     arr.append(Balance)
#     arr.append(recall)
#     arr.append(pf)
#     arr.append(precision)

#     print(arr)

#     scores.loc[last] = arr
#     last = last + 1

# scores.to_csv('/home/cloudam/qt/test/test5.csv')

