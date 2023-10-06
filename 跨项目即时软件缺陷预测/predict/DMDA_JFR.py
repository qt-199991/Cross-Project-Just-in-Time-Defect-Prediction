import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from pofb20 import get_pofb20, get_efforts
from scipy.linalg import norm, block_diag
from sklearn import linear_model, neighbors, svm, naive_bayes, tree, ensemble, neural_network

dataset = np.asarray(['broadleaf','go','nova', 'openstack','platform','qt','gerrit','matplotlib','brackets','camel'])
# dataset = np.asarray([ 'go','matplotlib', 'openstack','broadleaf','nova', 'platform','qt','gerrit','brackets','camel'])
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


def DMDA_JFR(X_src,Y_src, X_tar, parameter, r, classifier):

    Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r, classifier)

    for l in range(parameter["layers"]):
        if len(np.unique(Y_tar_pseudo)) == 2:
            num_sub_src, sub_src = subData(X_src, Y_src)
            num_sub_tar, sub_tar = subData(X_tar, Y_tar_pseudo)
            label = np.unique(Y_src)
            Num_Class = len(label)
            # // DA-GMDA stage
            # Construct matrix 𝜦 with Eq. (11);
            # Obtain the mapping matrix 𝐖 with Eq. (10);
            # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠 );
            # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡 );
            X_src_globalx, X_tar_globalx, W = DA_GMDA(X_src, Y_src, X_tar, Y_tar_pseudo, parameter)
            # Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src)  # Update pseudo label

            # // DA-LMDA stage
            X_src_localx = []
            X_tar_localx = []
            for c in range(Num_Class):
                # Construct matrix 𝜦𝑐 with Eq. (14);
                # Obtain the local subset mapping matrix 𝐖𝑐 with Eq. (13);
                # Obtain the source local feature presentations 𝐗𝑙𝑠,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠,𝑐 );
                # Obtain the target local feature presentations 𝐗𝑙𝑡,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡,𝑐 );

                X_src_localxc, X_tar_localxc, Wc = DA_LMDA(Xc_src=sub_src[c], Yc_src=c * np.ones((num_sub_src[c], 1)),
                                                           Xc_tar=sub_tar[c], Yc_tar_pseudo=Y_tar_pseudo,
                                                           parameter=parameter)
                X_src_localx.append(X_src_localxc)
                X_tar_localx.append(X_tar_localxc)

                # Y_tar_pseudo = Pseudolabel(X_src_localx, X_tar_localx, Y_src)  # Update pseudo label

            X_src_localx = np.concatenate((X_src_localx[0], X_src_localx[1]), axis=0)
            X_tar_localx = np.concatenate((X_tar_localx[0], X_tar_localx[1]), axis=0)

            # The source features presentations [𝐗𝑙𝑠,𝐗𝑙𝑠,𝑐1 ,𝐗𝑙𝑠,𝑐2 ];
            X_src_new = np.concatenate((X_src_globalx, X_src_localx),
                                       axis=1)  # X_src_new: source joint feature representations
            # The target features presentations [𝐗𝑙𝑡,𝐗𝑙𝑡,𝑐1 ,𝐗𝑙𝑡,𝑐2 ];
            X_tar_new = np.concatenate((X_tar_globalx, X_tar_localx),
                                       axis=1)  # X_tar_new: target joint feature representations

            # Train a classifier 𝑓(⋅) by using the source features presentations and labels of source project;
            Y_tar_pseudo_New = Pseudolabel(X_src_new, X_tar_new, Y_src, r, classifier)

            if len(np.unique(Y_tar_pseudo)) == 2:
                Y_tar_pseudo = Y_tar_pseudo_New
            else:
                pass
        else:
            Y_tar_pseudo = Pseudolabel(X_src, X_tar, Y_src, r+100, classifier)


    return Y_tar_pseudo

def DA_GMDA(X_src,Y_src,X_tar,Y_tar_pseudo, parameter):

    # Construct matrix 𝜦 with Eq. (11);
    n_src = X_src.shape[0]  # instance number
    n_tar = X_tar.shape[0]  # instance number
    X = np.concatenate((X_src.T, X_tar.T), axis=1)  # concat in columns
    m = X.shape[0]  # row number: feature number
    n = X.shape[1]  # column number: instance number
    if n != (n_src + n_tar):
        print("Concact matrix is wrong, please check!")
    label = np.unique(Y_src)
    Num_Class = len(label)
    # Num_Class = len(np.unique(Y_src))

    # X = X * np.diag(sparse(1. / np.sqrt(np.sum(np.square(X), 1))))
    X_normalize = preprocessing.normalize(X, axis=0, norm='l2')
    # X = np.c_(X_normalize,  np.ones((1, n)))
    # X = np.insert(X_normalize, -1, values=np.ones((1, n)), axis=0)
    X = np.row_stack((X_normalize, np.ones((1, n))))  # adding bias
    Hsm = np.multiply(1 / np.square(n_src), (np.ones((n_src, 1)) * np.ones((n_src, 1)).T))
    Htm = np.multiply(1 / np.square(n_tar), (np.ones((n_tar, 1)) * np.ones((n_tar, 1)).T))
    Hstm = np.multiply(1 / (n_tar * n_src), (np.ones((n_src, 1)) * np.ones((n_tar, 1)).T))  # ? -

    num_sub_src, sub_src = subData(X_src, Y_src)
    num_sub_tar, sub_tar = subData(X_tar, Y_tar_pseudo)

    tempHsc = []
    tempHtc = []
    tempHstc = []
    for i in range(Num_Class):
        n_src_c = num_sub_src[i]  # ns,c
        Hsc1 = np.multiply(1 / np.square(n_src_c), (np.ones((n_src_c, 1)) * np.ones((n_src_c, 1)).T))
        n_tar_c = num_sub_tar[i]  # nt,c
        Htc1 = np.multiply(1 / np.square(n_tar_c), (np.ones((n_tar_c, 1)) * np.ones((n_tar_c, 1)).T))
        Hstc1 = np.multiply(1 / (n_tar_c * n_src_c), (np.ones((n_src_c, 1)) * np.ones((n_tar_c, 1)).T)) 

        tempHsc.append(Hsc1)
        tempHtc.append(Htc1)
        tempHstc.append(Hstc1)
    Hsc = block_diag(tempHsc[0], tempHsc[1])
    Htc = block_diag(tempHtc[0], tempHtc[1])
    Hstc = block_diag(tempHstc[0], tempHstc[1])


    # Mss = np.dot(np.dot(X_src.T, (Hsm + Hsc)), X_src)  #
    # Mst = np.dot(np.dot(X_src.T, (Hstc + Hstm)), X_tar)  #
    # Mtt = np.dot(np.dot(X_tar.T, (Htm + Htc)), X_tar)  #X_tar * (Htm + Htc) * X_tar.T  # ? in the original paper is wrong: X_tar.T * (Hsm + Hsc) * X_tar
    #
    # A = np.concatenate((np.concatenate((Mss, -Mst), axis=1), np.concatenate((-Mst, Mtt), axis=1)), axis=0)
    # A = A / np.sqrt(np.sum(np.diag(A.T * A)))  # normalize A, the Frobenius norm, sqrt(sum(diag(X'*X))).
    # S2 = np.dot(A, A.T)  # A.T * A   # MMD

    M = np.concatenate((np.concatenate(((Hsm + Hsc), -(Hstc + Hstm)), axis=1),
                        np.concatenate((-(Hstc.T + Hstm.T), (Htm + Htc)), axis=1)), axis=0)
    M = M / np.sqrt(np.sum(np.diag(M.T * M)))
    S2 = np.dot(np.dot(X, M), X.T)  # MMD
    S1 = np.dot(X, X.T)  # or X * X.T  # scatter matrix

    # Obtain the mapping matrix 𝐖 with Eq. (10);
    q = np.ones((m+1, 1)) * (1-parameter["noises"])
    q[-1] = 1
    EQ1 = np.multiply(S1, (q * q.T))
    np.fill_diagonal(EQ1, np.multiply(q, np.diag(S1)))
    # EQ1[1:m + 2:end] = np.multiply(q, np.diag(S1))  # diag 对角线
    EQ2 = np.multiply(S2, (q * q.T))
    np.fill_diagonal(EQ2, np.multiply(q, np.diag(S2)))
    # EQ2[1:m + 2:end] = np.multiply(q, np.diag(S2))
    EP = np.multiply(q, S1)  #  EP = np.multiply(Sc[:-1, :], repmat(q.T, m, 1))  # m*(m+1)
    reg = parameter["lambda"] * np.eye((m + 1))
    reg[-1, -1] = 0

    W = EP / (EQ1 + reg + parameter["beta"] * EQ2)  # global feature  matrix mapping??"beta"

    # Obtain the source global feature presentations 𝐗𝑙𝑠 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠);
    global_all = np.dot(W, X) #?? W*X.T
    global_all = np.tanh(global_all)
    global_all = preprocessing.normalize(global_all, axis=0, norm='l2')
    # global_all = global_all * np.diag(sparse(1. / sqrt(np.sum(np.square(global_all)))))  # ?? WHY?
    X_src_globalx = global_all[:, :n_src].T  # source global feature representations


    # Obtain the target global feature presentations 𝐗𝑙𝑡= 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡);
    X_tar_globalx = global_all[:, n_src:].T  # target global feature representations

    # Y_tar_pseudo = Pseudolabel(X_src_globalx, X_tar_globalx, Y_src)  # Update pseudo label
    # X = global_all

    return X_src_globalx, X_tar_globalx, W

def DA_LMDA(Xc_src, Yc_src, Xc_tar, Yc_tar_pseudo, parameter):


    # Construct matrix 𝜦𝑐 with Eq. (14);
    label = np.unique(Yc_src)
    # Num_Class = len(label)
    Xc = np.concatenate((Xc_src.T, Xc_tar.T), axis=1)  # concat in rows
    m = Xc.shape[0]  # row number: feature number
    nc = Xc.shape[1]  # column number: instance number
    n_src_c = len(Xc_src)
    n_tar_c = len(Xc_tar)
    if nc != (n_src_c + n_tar_c):
        print("Concact matrix is wrong, please check!")

    # 向量化（右乘对角矩阵）,采用右乘一个对角矩阵的方式对矩阵进行缩放，常见的列向量单位化操作。2-范数：║x║2=（│x1│2+│x2│2+…+│xn│2）1/2
    # Xc = Xc * np.diag(sparse(1. / np.sqrt(np.sum(np.square(Xc), 1))))
    Xc_normalize = preprocessing.normalize(Xc, axis=0, norm='l2')
    # Xc = np.concatenate((Xc_normalize, np.ones(1, nc)), axis=1)
    Xc = np.row_stack((Xc_normalize, np.ones((1, nc))))

    Hsc_c = np.multiply((1 / np.square(n_src_c)), (np.ones((n_src_c, 1)) * np.ones((n_src_c, 1)).T))

    Htc_c = np.multiply((1 / np.square(n_tar_c)), (np.ones((n_tar_c, 1)) * np.ones((n_tar_c, 1)).T))
    Hstc_c = np.multiply((1 / (n_tar_c * n_src_c)), (np.ones((n_src_c, 1)) * np.ones((n_tar_c, 1)).T))

    # Mss_c = np.dot(np.dot(Xc_src.T, Hsc_c), Xc_src)
    # Mst_c = np.dot(np.dot(Xc_src.T, Hstc_c), Xc_tar)
    # Mtt_c = np.dot(np.dot(Xc_tar.T, Htc_c), Xc_tar)

    # Ac = np.concatenate((np.concatenate((Mss_c, -Mst_c), axis=1), np.concatenate((-Mst_c, Mtt_c), axis=1)), axis=0)
    # Ac = Ac / np.sqrt(np.sum(np.diag(Ac.T * Ac)))  # normalize Ac
    # S2c = Ac * Ac.T  # MMD

    Mc = np.concatenate((np.concatenate((Hsc_c, -Hstc_c), axis=1), np.concatenate((-Hstc_c.T, Htc_c), axis=1)), axis=0)
    Mc = Mc / np.sqrt(np.sum(np.diag(Mc.T * Mc)))  # MMD
    S2c = np.dot(np.dot(Xc, Mc), Xc.T)

    S1c = np.dot(Xc, Xc.T)

    # Obtain the local subset mapping matrix 𝐖𝑐 with Eq. (13);
    q = np.ones((m + 1, 1)) * (1 - parameter["noises"])
    EQ1c = np.multiply(S1c, (q * q.T))
    np.fill_diagonal(EQ1c, np.multiply(q, np.diag(S1c)))   # diag 对角线
    EQ2c = np.multiply(S2c, (q * q.T))
    np.fill_diagonal(EQ2c, np.multiply(q, np.diag(S2c)))   # diag 对角线
    EPc = np.multiply(q, S1c)  # EPc = np.multiply(Sc[:-1, :], repmat(q.T, m, 1))  # m*(m+1)
    reg = parameter["lambda"] * np.eye((m + 1))
    reg[-1, -1] = 0
    Wc = EPc / (EQ1c + reg + parameter["beta"] * EQ2c)
    has_nan = np.isnan(Wc)
    if has_nan.any():
        print("Wc存在NaN值**********")
        column_means = np.nanmean(Wc, axis=0)
        for i in range(Wc.shape[0]):
            for j in range(Wc.shape[1]):
                if np.isnan(Wc[i, j]):
                    Wc[i, j] = column_means[j]
    # Obtain the source local feature presentations 𝐗𝑙𝑠,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑠,𝑐 );
    local_all = np.dot(Wc, Xc)  # local feature  matrix mapping
    local_all = np.tanh(local_all)
    local_all = preprocessing.normalize(local_all, axis=0, norm='l2')
    # local_all = local_all * diag(sparse(1. / np.sqrt(np.sum(np.square(local_all)))))

    X_src_localx = local_all[:, :n_src_c].T  # source local feature representations

    # Obtain the target local feature presentations 𝐗𝑙𝑡,𝑐 = 𝑡𝑎𝑛ℎ(̃𝐗𝑙−1𝑡,𝑐 );
    X_tar_localx = local_all[:, n_src_c:].T  # target local feature representations

    return X_src_localx, X_tar_localx, Wc

def Pseudolabel(src_X, tar_X, src_labels, r, classifier="LR"):
  
    if classifier == "LR":
        # Logistic Regression classifier
        Y = src_labels
        # Y = (src_labels + 1) * 0.5 # [-1,1] vs[0,1]
        lr = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr',class_weight = 'balanced')
        # lr = linear_model.LogisticRegression(solver='liblinear', max_iter=100, multi_class='ovr')
        # print("src_X",src_X)
        # print("Y", Y)
        # print('Y', Counter(Y))
        lr.fit(src_X, Y)
        predlabel = lr.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(lr.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if lr.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_pseudo = (prob_pos >= 0.5).astype(np.int32)
        Y_tar_pseudo = Y_pseudo
        # has_nan = np.isnan(Y_tar_pseudo)

        # if np.any(has_nan):
        #     print("矩阵含有NaN值********************")
        # else:
        #     print("矩阵不含有NaN值")
        # print("Y_tar_pseudo", Counter(Y_tar_pseudo))
        # print("Y_tar_pseudo",type(Y_tar_pseudo))
        # print("Y_tar_pseudo", Y_tar_pseudo)
        # Y_tar_pseudo = Y_pseudo * 2 - 1

    elif classifier == "KNN":
        # %% K-Nearest Neighbor (KNN) classifier
        knn = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                             metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                                             weights='uniform')  # IB1 = knn IB1（KNN,K=1）
        knn.fit(src_X, src_labels)
        Y_tar_pseudo = knn.predict(tar_X)

        # knn_model=fitcknn(src_X,src_labels,'NumNeighbors',1)
        # Y_tar_pseudo=knn_model.predict(tar_X)

    elif classifier == "SVM":
        svml = svm.SVC(probability=True, 
                   class_weight = 'balanced')  # SVM + Linear  Kernel (SVML)
        svml.fit(src_X, src_labels)
        predlabel = svml.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(svml.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if svml.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

    elif classifier == "RF":
        # Random Forest (RF)
        rf = ensemble.RandomForestClassifier(n_estimators=100)
        rf.fit(src_X, src_labels)
        predlabel = rf.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(rf.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if rf.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

    elif classifier == "NB":
        # Naive Bayes(NB)
        gnb = naive_bayes.GaussianNB(priors=None)
        gnb.fit(src_X, src_labels)
        predlabel = gnb.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(gnb.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if gnb.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_tar_pseudo = (prob_pos >= 0.5).astype(np.int32)

    else:
        # Logistic Regression classifier
        # Y = (src_labels + 1) * 0.5  # [-1,1] vs[0,1]
        Y = src_labels
        lr = linear_model.LogisticRegression(solver='liblinear',
                                             max_iter=100, multi_class='ovr', class_weight = 'balanced')
        # lr = linear_model.LogisticRegression(solver='liblinear',
                                            #  max_iter=100, multi_class='ovr')
        lr.fit(src_X, Y)
        predlabel = lr.predict_proba(tar_X)  # predict probability
        prob_pos = []  # the probability of predicting as positive
        prob_neg = list()  # the probability of predicting as negative
        for j in range(predlabel.shape[0]):  # 每个类的概率, array类型
            if len(lr.classes_) <= 1:
                print('第%d个样本预测出错' % j)
            else:
                if lr.classes_[1] == 1:
                    prob_pos.append(predlabel[j][1])
                    prob_neg.append(predlabel[j][0])
                else:
                    prob_pos.append(predlabel[j][0])
                    prob_neg.append(predlabel[j][1])
        prob_pos = np.array(prob_pos)
        Y_pseudo = (prob_pos >= 0.5).astype(np.int32)

        # model = glmfit(src_X,Y,'binomial', 'link', 'logit')
        # predlabel = glmval(model,tar_X, 'logit')
        # Y_pseudo=double(predlabel>=0.5)
        Y_tar_pseudo = Y_pseudo
        # Y_tar_pseudo = Y_pseudo * 2 - 1

    return Y_tar_pseudo

def subData(Xc, Y):
    num_sub = []  # [number of non-defective instances, number of defective instances]
    sub_X = []
    label = np.unique(Y)
    for i in range(len(label)):
        index = [j for j, x in enumerate(Y) if x == label[i]]
        num_sub.append(len(index))
        sub_X.append(Xc[index, :])

    return num_sub, sub_X

parameters = [[{"layers": 4, "noises": 0.6, "lambda": 0.01, "beta": 0.1},
                  {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},
                  {"layers": 4, "noises": 0.7, "lambda": 0.1, "beta": 0.1}],
                 [{"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},
                  {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1},
                  {"layers": 8, "noises": 0.6, "lambda": 0.01, "beta": 0.1}
                  ]]

last = 0
n = 0
for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    m = -1
    for j in range(len(dataset)):  # 遍历数据库，选定源项目
        m = m + 1
        n = n + 1
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
            # print('y_sample', Counter(y_sample))
            # print('y_target', Counter(y_target))
            sc = StandardScaler()
            X_sample = sc.fit_transform(X_sample)
            X_target = sc.transform(X_target)

            # imb = SMOTE()
            # X_sample, y_sample = imb.fit_resample(X_sample, y_sample)
            y_predict = DMDA_JFR(X_sample, y_sample, X_target, parameters[0][0], r, classifier='RF')
            # has_nan = np.isnan(y_predict)

            # if np.any(has_nan):
            #     print("y_predict矩阵含有NaN值********************")
            # else:
            #     print("y_predict矩阵不含有NaN值")

            # print("y_predict", y_predict)
            # print('y_predict', Counter(y_predict))
            pofb20 = get_pofb20(np.array(y_target), y_predict, efforts)
            # print(y_pred)
            arr.append(round(pofb20 * 100, 2))
            # auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_predict)
            # arr.append(round(auc * 100, 2))
            # arr.append(round(f1_score(y_target, y_predict) * 100, 2))
            # arr.append(round(matthews_corrcoef(y_target, y_predict) * 100, 2))
            # arr.append(round(GMean * 100, 2))
            # arr.append(round(Balance * 100 , 2))
            # arr.append(round(recall * 100, 2))
            # arr.append(round(pf * 100, 2))
            # arr.append(round(precision * 100, 2))
            print(arr)

            scores.loc[last] = arr
            last = last + 1

scores.to_csv('/home/cloudam/qt/pofb20/DMDA_JFR_pofb20_rf.csv')

# last = 0
# j = 3  #训练
# i = 4  #测试
# for r in range(20,70):
#     arr = []
#     arr.append(dataset[j])
#     print(dataset[j])
#     arr.append(dataset[i])
#     print(dataset[i])
#     X_sample, y_sample = trainMaker(dataset[j], r)
#     X_target, y_target = datasetMaker(dataset[i])
#     print('y_sample', Counter(y_sample))
#     # print('y_target', Counter(y_target))
    
#     # sc = StandardScaler()
#     # X_sample = sc.fit_transform(X_sample)
#     # X_target = sc.transform(X_target)

#     y_predict = DMDA_JFR(X_sample, y_sample, X_target, parameters[0][0], r, classifier='LR')
#     print('y_predict', Counter(y_predict))
#     auc, Balance, GMean, recall, pf, precision = GMean_and_Balance(y_target, y_predict)
#     arr.append(auc)
#     arr.append(f1_score(y_target, y_predict))
#     arr.append(matthews_corrcoef(y_target, y_predict))
#     arr.append(GMean)
#     arr.append(Balance)
#     arr.append(recall)
#     arr.append(pf)
#     arr.append(precision)

#     print(arr)

#     scores.loc[last] = arr
#     last = last + 1





