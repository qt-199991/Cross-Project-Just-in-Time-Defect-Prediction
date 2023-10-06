#coding=utf-8
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
np.set_printoptions(threshold=np.inf)
dataset = ['brackets.arff', 'broadleaf.arff', 'camel.arff', 'fabric.arff', 'jgroups.arff',
           'matplotlib.arff', 'neutron.arff', 'nova.arff',
           'npm.arff', 'sentry.arff', 'spring-integration.arff', 'tomcat.arff', 'wagtail.arff', 'zulip.arff']
columns=['fix', 'ns', 'nd', 'nf', 'entrophy ', 'la', 'ld', 'lt', 'ndev ', 'age', 'nuc', 'exp', 'rexp', 'sexp', 'bug',' author_date_unix_timestamp numeric','commit_type numeric']
def arff_to_csv(fpath):
    #读取arff数据
    if fpath.find('.arff') <0:
        print('the file is nott .arff file')
        return
    f = open(fpath)
    lines = f.readlines()
    lines = lines[20:]
    content = []
    for l in lines:
        content.append(l)
    datas = []
    for c in content:
        cs = c.split(',')
        datas.append(cs)

    #将数据存入csv文件中
    df = pd.DataFrame(data=datas,index=None,columns=columns)
    df['-'] = '-'
    df['--'] = '--'
    col_name = df.columns.values
    cols = list(df)
    cols.insert(0, cols.pop(cols.index(' author_date_unix_timestamp numeric')))
    cols.insert(1, cols.pop(cols.index('commit_type numeric')))
    # cols.insert(17, cols.pop(cols.index('fix')))
    # cols.insert(7, cols.pop(cols.index('nf')))
    # cols.insert(6, cols.pop(cols.index('nd')))
    # cols.insert(5, cols.pop(cols.index('ns')))
    cols.insert(2, cols.pop(cols.index('--')))
    cols.insert(3, cols.pop(cols.index('bug')))
    cols.insert(4, cols.pop(cols.index('-')))
    cols.insert(12, cols.pop(cols.index('fix')))
    # cols.insert(3, cols.pop(cols.index('bug')))
    df = df.loc[:, cols]
    df.replace('False', int(0), inplace=True)
    df.replace('FALSE', int(0), inplace=True)
    df.replace('TRUE', int(1), inplace=True)
    df.replace('True', int(1), inplace=True)
    # df.to_csv('E:/14_dataset/data10/brackets.csv')
    filename = fpath[:fpath.find('.arff')] + '.csv'
    df = df[:-1]
    df.to_csv(filename,index=None)
for i in range(len(dataset)):
    path = 'E:/14_dataset/data10/' + dataset[i]
    arff_to_csv(path)

# def changecolumn(file):
#     df = pd.read_csv(file)
#     df['-'] = '-'
#     col_name = df.columns.values
#     cols = list(df)
#     cols.insert(0, cols.pop(cols.index(' author_date_unix_timestamp numeric')))
#     cols.insert(1, cols.pop(cols.index('commit_type numeric')))
#     # cols.insert(17, cols.pop(cols.index('fix')))
#     # cols.insert(7, cols.pop(cols.index('nf')))
#     # cols.insert(6, cols.pop(cols.index('nd')))
#     # cols.insert(5, cols.pop(cols.index('ns')))
#     cols.insert(2, cols.pop(cols.index('bug')))
#     cols.insert(3, cols.pop(cols.index('-')))
#     cols.insert(11, cols.pop(cols.index('fix')))
#     # cols.insert(3, cols.pop(cols.index('bug')))
#     df = df.loc[:, cols]
#     df.to_csv('E:/14_dataset/data10/brackets.csv')
#
#
# changecolumn('E:/14_dataset/data10/brackets.csv')  #之后直接在csv文件中直接替换

# def replace(file):
#     df = pd.read_csv(file)
#     # print(df)
#     df.replace('False', int(0), inplace=True)
#     df.replace('FALSE', int(0), inplace=True)
#     df.replace('TRUE', int(1), inplace=True)
#     df.replace('True', int(1), inplace=True)
#     df['bug'] = df['bug'].replace({'False': int(0), 'True': int(1)})
#     # df['fix'] = df['fix'].map(change)
#     df.to_csv('E:/14_dataset/data10/brackets.csv')
# replace('E:/14_dataset/data10/brackets.csv')

# df = pd.read_csv('E:/14_dataset/data10/brackets.csv')
# print(df)