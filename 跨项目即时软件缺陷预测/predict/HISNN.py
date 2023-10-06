import numpy as np
import pandas as pd
import math
from sklearn import svm
from sklearn.utils import shuffle
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, linear_model
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.linalg import det
import copy
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
import time
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import os
from pofb20 import get_pofb20, get_efforts
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt','gerrit','matplotlib','brackets','camel'])

# scores = pd.DataFrame(columns=['Source', 'Target', 'AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision'])
scores = pd.DataFrame(columns=['Source','Target', 'pofb20'])
def mahal(Y, X):
    '''
     度量一个样本点y与数据分布为X的集合的距离
    :param Y: 观测数据 Y=[y1,y2,...,ym], Y为m×n的矩阵，m为样本数，n为特征数，y为一个向量1×n
    :param X: 样本数据 X = [x1,x2,...,xk]，X为k×n的矩阵，k为样本数，n为特征数，x为一个向量1×n
    :return: dist: list, Y中每个样本到集合X的马氏距离。
    '''
    ry = Y.shape[0]
    cy = Y.shape[1]
    u = np.mean(X, axis=0)  # mean value of samples  求所有样本每个属性的均值
    # print('u:', u)
    XT = X.T
    sigma = np.cov(XT)  # covariance matrix: 特征和特征之间的
    # print('sigma:', sigma, sigma.shape)
    try:
       sigma_inv = np.linalg.inv(sigma)  # the inverse of covariance matrix
       # print('s_i:', np.linalg.inv(sigma))
    except:
        print("Inverse matrix does not exist!")
        raise ValueError('Singular matrix')
    dist = []
    for i in range(len(Y)):
        y = Y[i, :]   # y-u：[0.3 3.1 2.9] (3,)
        delta = (y - u).reshape(1, -1)  # delta: [[0.3 3.1 2.9]] (1,3)
        # print('delta:', delta, delta.shape)
        di = np.dot(np.dot(delta, sigma_inv), (delta.T) )
        # print('di:', di)
        di = di.tolist()
        dist.append(di[0][0])

    return dist

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
        # typicalFracDict = {1: 0.9, 0: 0.9}
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

def HISNN_train(source, target, K=10):

    target_x = target[:, : - 1]
    source_x = source[:, : - 1]

    if np.abs(det(np.cov(source_x.T))) >= 10e-6:  #返回矩阵的行列式1

        Dsms = mahal(source_x, source_x)  #返回的是不一样的值

        # print('Dsms1:', len(Dsms))  # Dsms,
    else:
        Dsms = pdist(np.vstack((np.mean(source_x, axis=0), source_x)), 'euclidean')   # 垂直拼接
        # print('Dsms2:', len(Dsms))    #Dsms 57630

    Dsms = Dsms[: len(source_x)]  #只取均值与其他几行的距离，就是样本的个数
    std1 = np.std(Dsms) #是一个数字
    idx_outlier1 = []
    for j in range(len(Dsms)):
        if Dsms[j] > 3 * std1:
            #返回一个列向量，表示向量中所有非零元素的索引。
            idx_outlier1.append(j)  #  find(vector) - Return a column vector denoting the index of all non - zero elements in vector.

    if det(np.cov(target_x.T)) != 0:  # and (mt_rank == target_x.shape[1])
        # Mahalanobis distance
        Dtms = mahal(source_x, target_x)

    else:
        Dtms = pdist(np.vstack((np.mean(target_x, axis=0), source_x)), 'euclidean')

    Dtms = Dtms[: len(source_x)] #返回的是目标数据均值距离每一个源样本的距离
    std2 = np.std(Dtms)
    idx_outlier2 = []
    for l in range(len(Dtms)):
        if Dtms[l] > 3 * std2:
            idx_outlier2.append(l)

    if len(idx_outlier2) > 0.8 * len(source):
        # In some case, len(idx_outlier2) == len(source), 如果是这样1，则所有源数据都是离群值，即没有训练数据。
        idx_outlier2 = []

    idx_outlier1.extend(idx_outlier2)  #用于在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
    idx_outlier = np.unique(idx_outlier1)
    # print('idx_outlier:', len(idx_outlier))  #  idx_outlier,
    index = list(range(len(source)))
    idx_inlier = list(set(index) - set(idx_outlier))  # inlier index

    inds = []
    da = np.zeros((len(source_x), len(target_x)))  # 初始化测试实例到每个源实例的距离。
    for i in range(len(target_x)):
        for j in range(len(source_x)):   # each instance in source data
            # Calculate hamming distance between a target instance and a source instance
            da[j, i] = pdist(np.vstack((target_x[i, :], source_x[j, :])), 'hamming')
        ind = da[:, i].argsort().tolist()  # ascending order 数组升序排序返回索引，并数组转换为列表
        inds.extend(ind[: K])  # 选择距离最近的k个值的索引列表
    inds.extend(idx_inlier)  # len(inds+inlier)
    new_source = source[np.unique(inds), :]  # Remove the duplicated instances
    return new_source

last = 0
for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    for j in range(len(dataset)):  # 遍历数据库，选定源项目
        for r in range(1):
            if i == j: continue
            arr = []
            arr.append(dataset[j])
            print(dataset[j])
            arr.append(dataset[i])
            print(dataset[i])
            efforts = get_efforts(dataset[i])
            X_sample, y_sample = trainMaker(dataset[j], r)
            X_target, y_target = datasetMaker(dataset[i])

            sc = StandardScaler()
            X_sample = sc.fit_transform(X_sample)
            X_target = sc.transform(X_target)
        
            source = np.c_[X_sample, y_sample]
            target = np.c_[X_target, y_target]

            source_HISNN = HISNN_train(source, target, K=10)
            # clf = GaussianNB()
            # clf = GaussianNB(priors=None)
            # clf = ensemble.RandomForestClassifier(n_estimators=100)
            clf = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')
            # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
            # clf = svm.SVC()
            clf.fit(source_HISNN[:, : -1], source_HISNN[:, -1])
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
scores.to_csv('/home/cloudam/qt/pofb20/HISNN_pofb20_lr.csv')

# last = 0
# j = 5  #训练
# i = 4  #测试
# for r in range(20,40):
#     arr = []
#     arr.append(dataset[j])
#     print(dataset[j])
#     arr.append(dataset[i])
#     print(dataset[i])
#     X_sample, y_sample = trainMaker(dataset[j], r)
#     X_target, y_target = datasetMaker(dataset[i])
#     print('y_sample', Counter(y_sample))

#     sc = StandardScaler()
#     X_sample = sc.fit_transform(X_sample)
#     X_target = sc.transform(X_target)

#     source = np.c_[X_sample, y_sample]
#     target = np.c_[X_target, y_target]
#     source_HISNN = HISNN_train(source, target, K=10)
#     print('source_HISNN[:, -1]', Counter(source_HISNN[:, -1]))
#     print('source_HISNN',source_HISNN)

#     # clf = svm.SVC(class_weight = 'balanced')
#     # clf = GaussianNB(priors=None)
#     # clf = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear', max_iter=100, multi_class='ovr')
#     # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
#     clf = ensemble.RandomForestClassifier(n_estimators=100)

#     clf.fit(source_HISNN[:, : -1], source_HISNN[:, -1])
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

# scores.to_csv('/home/cloudam/qt/test/test4.csv')

