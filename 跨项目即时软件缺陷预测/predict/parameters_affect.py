from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from collections import Counter
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")
import math
import numpy as np
from sklearn.utils import shuffle
import sklearn
import pandas as pd
from cvxopt import matrix, solvers
from sklearn import preprocessing
import random

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


def similarityWeight(X_sample, X_target):  # 这是相似度权重，求每个样本与目标样本的相似度
    max_elem = np.amax(X_target, axis=0)
    min_elem = np.amin(X_target, axis=0)

    similarityValue = []

    s_max = 0

    for i in range(X_sample.shape[0]):  # 遍历每个样本的每列属性
        s = 0
        for j in range(X_sample.shape[1]):
            if min_elem[j] <= X_sample[i, j] <= max_elem[j]:
                s += 1

        s = s / X_sample.shape[1]  # 这是属性的个数
        s_max = max(s_max, s)
        similarityValue.append(s)

    similarityWeights = np.array(similarityValue) / s_max
    return similarityWeights



def datasetMaker(dataset_name):
    path = 'F:/朱老师课题/自己的代码/研究点一/data/' + dataset_name + '.csv'
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

def kernel(ker, X1, X2, gamma):
    K = None
    if ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1), np.asarray(X2))
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1))
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), np.asarray(X2), gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1), None, gamma)
    return K

class KMM:
    def __init__(self, kernel_type='linear', gamma=1.0, B=1.0, eps=None):
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.B = B
        self.eps = eps

    def fit(self, Xs, Xt):
        ns = Xs.shape[0]
        nt = Xt.shape[0]
        if self.eps == None:
            self.eps = self.B / np.sqrt(ns)
        K = kernel(self.kernel_type, Xs, None, self.gamma)
        kappa = np.sum(kernel(self.kernel_type, Xs, Xt, self.gamma) * float(ns) / float(nt), axis=1)

        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, ns)), -np.ones((1, ns)), np.eye(ns), -np.eye(ns)])
        h = matrix(np.r_[ns * (1 + self.eps), ns * (self.eps - 1), self.B * np.ones((ns,)), np.zeros((ns,))])

        sol = solvers.qp(K, -kappa, G, h)
        beta = np.array(sol['x'])
        return beta

def cumulativeWeight(weights):  #累计的权重
    p = weights.copy()
    p = np.cumsum(p)  #累加权重
    u = random.random() * p[-1]
    index = 0
    for index in range(len(p)):  #找出累加之后的权重开始大于u的索引
        if p[index] > u:
            break
    return index

def BSW(X_sample, y_sample, sample_beta, Psize):
    w_0 = []
    w_1 = []

    Psize = int(Psize * len(y_sample))
    print('Psize',Psize)

    X_final = []
    y_final = []

    X_0 = []
    X_1 = []

    for i in range(len(y_sample)):
        if y_sample[i] == 1:
            X_1.append(X_sample[i])
            w_1.append(sample_beta[i])
        else:
            X_0.append(X_sample[i])
            w_0.append(sample_beta[i])

    w_0 = np.asarray(w_0)
    w_1 = np.asarray(w_1)

    X_0 = np.asarray(X_0)
    X_1 = np.asarray(X_1)

    for i in range(Psize):   #在没有缺陷的里面抽取
        index = cumulativeWeight(w_0)
        X_final.append(X_0[index])
        y_final.append(0)

    for i in range(Psize): #在有缺陷的里抽取
        index = cumulativeWeight(w_1)
        X_final.append(X_1[index])
        y_final.append(1)

    X_final = np.asarray(X_final)
    y_final = np.asarray(y_final)

    X_final, y_final = shuffle(X_final, y_final, random_state=22)
    counts = np.bincount(y_final)  #这里面没用用到上采样
    # print('counts',counts)

    return X_final, y_final
def trainMaker(dataset_name, r):
    path = 'F:/朱老师课题/自己的代码/研究点一/data/' + dataset_name + '.csv'
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
        typicalFracDict = {1: 0.9, 0: 0.9}
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


def dataSelector(X_sample, y_sample, sample_beta, k):
    sorted_id = sorted(range(len(sample_beta)), key=lambda x: sample_beta[x], reverse=True)
    # print('sorted_id',sorted_id)
    X_sample_1 = np.array(X_sample)
    y_sample_1 = np.array(y_sample)
    sample_beta = np.array(sample_beta)
    # print('X_sample',X_sample[1189])
    l1 = Counter(y_sample)[1]
    l1 = l1 * k
    l0 = Counter(y_sample)[0]
    l0 = l0 * k
    X_sample = X_sample_1[sorted_id]  #这是拍好序的样本
    y_sample = y_sample_1[sorted_id]
    sample_beta = sample_beta[sorted_id]
    X_final = []
    y_final = []
    sample_beta_new = []

    j = 0
    h = 0

    for i in range(len(y_sample)):
        if y_sample[i] == 1:
            if j < l1:
                X_final.append(X_sample[i])
                y_final.append(y_sample[i])
                sample_beta_new.append(sample_beta[i])
                j = j + 1
        else:
            if h < l0:
                X_final.append(X_sample[i])
                y_final.append(y_sample[i])
                sample_beta_new.append(sample_beta[i])
                h = h + 1
    X_final = np.asarray(X_final)
    y_final = np.asarray(y_final)
    sample_beta_new = np.asarray(sample_beta_new)
    X_final, y_final, sample_beta_new = shuffle(X_final, y_final, sample_beta_new, random_state=22)

    return X_final, y_final, sample_beta_new

alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] #保留多少数据
psize = [0.0]  #重采样占总过滤之后数据的比例，这里采样之后取的是2*psize的数据
dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt'])

for a in alphas:
    for p in psize:
        last = 0
        scores = pd.DataFrame(columns=['Source','Target', 'alphas','psize','AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision'])
        for i in range(len(dataset)):  # 遍历数据库，选定目标项目
            for j in range(len(dataset)):  # 遍历数据库，选定源项目
                for r in range(5):
                    if i == j: continue
                    arr = []
                    arr.append(dataset[j])
                    print(dataset[j])
                    arr.append(dataset[i])
                    print(dataset[i])
                    X_sample, y_sample = trainMaker(dataset[j], r)
                    X_target, y_target = datasetMaker(dataset[i])
                    sc = StandardScaler()
                    X_sample = sc.fit_transform(X_sample)
                    X_target = sc.transform(X_target)
                    kmm = KMM(kernel_type='rbf', B=1)
                    sample_beta = kmm.fit(X_sample, X_target)
                    sample_beta = [i[0] for i in sample_beta]  #每个样本的权重
                    # print('sample_beta',sample_beta)
                    X_sample, y_sample, sample_beta = dataSelector(X_sample, y_sample, sample_beta, k=a)
                    # X_sample, y_sample = BSW(X_sample, y_sample, sample_beta, Psize=p)

                    clf = svm.SVC(class_weight='balanced')
                    # clf = linear_model.LogisticRegression(random_state=r + 1, solver='liblinear', max_iter=100, multi_class='ovr')
                    clf.fit(X_sample, y_sample)

                    y_pred = clf.predict(X_target)
                    auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_pred)
                    arr.append(a)
                    arr.append(p)
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
        scores.to_csv('E:/360downloads/kwm-parameters/kmm_' + str(a) + '_BSW_' + str(p) + '.csv')
