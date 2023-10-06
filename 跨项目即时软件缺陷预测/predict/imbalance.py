import pandas as pd
import numpy as np
from collections import Counter

# dataset = np.asarray(['gerrit','go','jdt', 'openstack','platform','qt'])
# dataset = ['gerrit', 'go', 'jdt', 'openstack', 'platform', 'qt', 'brackets', 'broadleaf', 'camel', 'fabric', 'jgroups',
#            'matplotlib', 'neutron', 'nova',
#            'npm', 'sentry', 'spring-integration', 'tomcat', 'wagtail', 'zulip']
dataset = ['broadleaf','go','nova', 'openstack','platform','qt','gerrit','matplotlib','brackets','camel']
def datasetMaker(dataset_name):
    path = 'H:/朱老师课题/自己的代码/研究点一/data/' + dataset_name + '.csv'
    # path = 'E:/JIT/Data_Extraction/git_base/datasets/' + dataset_name + '/cross/' + dataset_name + '_k_feature.csv'
    df = pd.read_csv(path)
    col_name = df.columns.values
    X = df[col_name[5:]].values
    X = np.array(X)
    X = X[-5000:]
    y = df[col_name[3]].values
    y = np.array(y)
    y = y.astype(int)
    y = y[-5000:]
    # print(y)
    y[y > 0] = 1

    return X, y
imblance = []
for i in range(len(dataset)):  # 遍历数据库，选定目标项目
    X_sample, y_sample = datasetMaker(dataset[i])
    print(i, dataset[i],Counter(y_sample))
    clean = int(Counter(y_sample)[0])
    bug = int(Counter(y_sample)[1])
    # imblance.append(bug / (bug + clean))
    print(dataset[i], 'imblance',bug / (bug + clean))
# print(imblance)