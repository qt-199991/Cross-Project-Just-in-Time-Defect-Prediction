import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import linear_model,ensemble
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter

from sklearn.metrics import precision_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import warnings

# import tca

warnings.filterwarnings("ignore")
import math

# def run_cross_validation_models_smote(X, y, kfold_value):
#     score_list = []
#
#     kf = KFold(n_splits = kfold_value, random_state=None, shuffle=False)
#     # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
#     # clf = linear_model.LogisticRegression(random_state= 1, solver='liblinear', max_iter=100, multi_class='ovr')
#     # clf = ensemble.RandomForestClassifier(random_state= 1, n_estimators=100)
#     clf = svm.SVC()
#     for train_index, test_index in kf.split(X):
#         arr = []
#         imb = SMOTE()
#         X_sample, y_sample = imb.fit_resample(X[train_index], y[train_index])
#         # l1 = Counter(y_sample)
#         # print(l1)
#         clf.fit(X_sample, y_sample)
#         predict = clf.predict(X[test_index])
#         auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(predict, y[test_index])
#         arr.append(auc)
#         arr.append(Balance)
#         arr.append(f1_score(predict, y[test_index]))
#         arr.append(GMean)
#         arr.append(matthews_corrcoef(predict, y[test_index]))
#         arr.append(pf)
#         arr.append(precision)
#         arr.append(recall)
#         score_list.append(arr)
#     score_list = np.array(score_list)
#     mean_score_arr= score_list.mean(axis=0)
#     return mean_score_arr

def GMean_and_Balance(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    auc = roc_auc_score(y_true, y_pred)

    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    pf = FP / (FP + TN)

    GMean = np.sqrt(recall * (1 - pf))

    Balance = 1 - (np.sqrt(pf ** 2 + (1 - recall) ** 2) / np.sqrt(2))

    return auc, Balance, GMean, recall, pf, precision

def datasetMaker(dataset_name):
    path = 'C:/Users/qiutian/Desktop/研究点一/data/' + dataset_name + '.csv'
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

def run_cross_validation_models(X, y, kfold_value):
    score_list = []

    kf = KFold(n_splits = kfold_value, random_state=None, shuffle=False)
    # clf = AdaBoostClassifier(base_estimator=GaussianNB(), n_estimators=20, learning_rate=0.1)
    # clf = svm.SVC(class_weight = 'balanced')
    clf = GaussianNB(priors=None)
    # clf = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr',class_weight = 'balanced')
    # clf = ensemble.RandomForestClassifier()

    for train_index, test_index in kf.split(X):
        arr = []
        clf.fit(X_sample, y_sample)
        predict = clf.predict(X[test_index])
        # print(predict)
        auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(predict, y[test_index])
        arr.append(auc)
        arr.append(f1_score(predict, y[test_index]))
        arr.append(matthews_corrcoef(predict, y[test_index]))
        arr.append(GMean)
        arr.append(Balance)
        arr.append(recall)
        arr.append(pf)
        arr.append(precision)
        score_list.append(arr)
        score_list.append(arr)
    score_list = np.array(score_list)
    mean_score_arr= score_list.mean(axis=0)
    return mean_score_arr

# dataset = np.asarray(['go','jdt', 'openstack','platform','qt'])
# dataset = np.asarray(['gerrit','go','jdt', 'openstack','platform','qt'])

dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt','gerrit','matplotlib','brackets','camel'])
scores = pd.DataFrame(columns=['Target', 'AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision'])
last = 0
for i in range(len(dataset)):
    # for j in range(20):
    arr = []
    arr.append(dataset[i])
    X_sample, y_sample = datasetMaker(dataset[i])
    score1 = run_cross_validation_models(X_sample, y_sample, 10)
    score1 = score1 * 100
    arr.extend(score1)
    scores.loc[last] = arr
    last = last + 1
    print(arr)
# scores.to_csv('F:/朱老师课题/自己的代码/研究点一/deal results/compare_intra_cross/intra_svc.csv',float_format='%.2f')
scores.to_csv('C:/Users/qiutian/Desktop/inter/intra_nb.csv',float_format='%.2f')