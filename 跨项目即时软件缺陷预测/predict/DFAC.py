import numpy as np
import math
import pandas as pd
from sklearn import linear_model,ensemble
import math
from sklearn.utils import shuffle
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
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

def DFAC(X_tr, y_tr, X_te, y_te):

    time_dicts = {}
    t1 = time.time()

    # step1：训练集度量元数据和测试集度量元数据合并
    X = np.vstack((X_te, X_tr))  # 垂直拼接
    y = np.hstack((y_te, y_tr))
    length_X = X.shape[0]
    length_te = X_te.shape[0]
    # length_tr = length_X - length_te
    # # X_tr_ori = X[length_te:]
    # # y_tr_ori = y[length_te:]

    # step2：聚类分簇
    c = int(np.floor(length_X / 10))
    # c = 10
    AC = AgglomerativeClustering(n_clusters=c, affinity='euclidean', memory=None,
                                 connectivity=None, compute_full_tree='auto', linkage='average',
                                 )  # 默认c=2
    AC.fit(X)
    labels = AC.labels_
    diff_labels = np.unique(labels)
    # print(labels, diff_labels)
    data = np.insert(X, -1, values=labels, axis=1)  # 末尾插入一列，为合并数据集添加簇标签
    length_data = data.shape[0]
    if length_data == length_X and data.shape[1] == X.shape[1] + 1:  # 判断合并数据是否正确
        pass
    else:
        print('数据合并报错')

    # step3：判断测试实例是否存在于簇中？存在，将训练集选出合并成训练集；不存在，删除没有测试实例存在的簇。
    del_index = []
    del_label = []
    retain_label = []
    retain_index = []
    for label in diff_labels:  # 注意len(labels) == X.shape[0], 因此必须设置为簇号列表，且簇号唯一
        index = np.where(data[:, -1] == label)[0].tolist()  # 簇号为label的样本索引(turple取值后为数组，再转为列表)
        # print(label, '\n', index)  # 检查簇及对应索引值
        index_test_instance = filter(lambda x: x < length_te, index)  # 筛选簇中索引为测试实例，返回filter
        index_test_instance = list(index_test_instance)  # 必须转为list才能使用, 簇中测试实例的索引列表
        index_train_instance = list(filter(lambda x: x >= length_te, index))

        if index_test_instance == list():
            # print('簇号为%d的数据无测试数据，删除该簇' % (label))
            del_index.extend(index)  # 列表插入数据，而不是列表和列表组合
            del_label.append(label)
        else:
            retain_label.append(label)
            retain_index.extend(index)
            # print('簇号为%d的数据中存在测试数据，因此,保留该簇' % (label))
    retain_label = np.unique(retain_label)  # 簇号去重
    del_label = set(del_label)  # 簇号去重
    # print('AC聚类删除的簇号为%s，保留的簇号为%s。\n' % (del_label, retain_label))
    X_new = np.delete(X, del_index, axis=0)  # 删除某些行
    y = np.ravel(y)
    y_new = np.delete(y, del_index, axis=0)
    X_tr_new = X_new[length_te:]
    y_tr_new = y_new[length_te:]

    if X_tr_new.shape[0] == y_tr_new.shape[0] \
            and X_tr_new.shape[0] + len(del_index) == X[length_te:].shape[0]:
        t2 = time.time()
        time_each = {str(DFAC): (t2 - t1)}
        time_dicts.update(time_each)
        df_filter_time = pd.Series(time_dicts)
    else:
        # print('删除训练样本出错，请检查！')
        pass

    # print('每个文件删除的样本行索引列表为 %s' % del_index)
    return df_filter_time, X_tr_new, y_tr_new
last = 0
for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    for j in range(len(dataset)):  # 遍历数据库，选定源项目
    # for j in range(1):
        for r in range(2):
            if i == j: continue
            arr = []
            arr.append(dataset[j])
            print(dataset[j])
            arr.append(dataset[i])
            print(dataset[i])
            efforts = get_efforts(dataset[i])
            X_sample, y_sample = trainMaker(dataset[j], r)
            X_target, y_target = datasetMaker(dataset[i])
            print('y_sample', Counter(y_sample))
            # sc = StandardScaler()
            # X_sample = sc.fit_transform(X_sample)
            # X_target = sc.transform(X_target)
           
            df_filter_time, X_sample, y_sample = DFAC(X_sample, y_sample, X_target, y_target)
            print('y_sample', Counter(y_sample))
            sc = StandardScaler()
            X_sample = sc.fit_transform(X_sample)
            X_target = sc.transform(X_target)
            clf = ensemble.RandomForestClassifier(n_estimators=100)
            # clf = svm.SVC()
            # clf = svm.SVC(class_weight = 'balanced')
            # clf = GaussianNB()
            # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
            # clf = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')
            imb = SMOTE()
            X_sample, y_sample = imb.fit_resample(X_sample, y_sample)
            clf.fit(X_sample,y_sample)
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
scores.to_csv('/home/cloudam/qt/pofb20/DFAC_pofb20_rf.csv')

# last = 0
# j = 0  #训练
# i = 5  #测试
# for r in range(30):
#     arr = []
#     arr.append(dataset[j])
#     print(dataset[j])
#     arr.append(dataset[i])
#     print(dataset[i])
#     X_sample, y_sample = trainMaker(dataset[j], r)
#     X_target, y_target = datasetMaker(dataset[i])
#     print('y_sample',Counter(y_sample))

#     sc = StandardScaler()
#     X_sampl = sc.fit_transform(X_sample)
#     X_target = sc.transform(X_target)
#     df_filter_time, X_sample_1, y_sample_1 = DFAC(X_sample, y_sample, X_target, y_target)


#     print('y_sample_1',Counter(y_sample_1))
#     clf = GaussianNB(priors=None)
#     clf.fit(X_sample_1,y_sample_1)
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


