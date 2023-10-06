#不同方法之间的条形图
import pandas as pd
import os
import matplotlib.pyplot as plt
# col_name = df.columns.values
# print(col_name)
# resultspath = 'F:/朱老师课题/自己的代码/研究点一/deal results/Mean_all/svc/'
resultspath = 'E:/360downloads/5000-2000到balance/meanall_measure/rf/'
csvfiles = os.listdir(resultspath)
print('csvfiles',csvfiles)
for i in range(len(csvfiles)):
    columnname = [csvfiles[i].rstrip(".csv")]
    df = pd.read_csv(resultspath + csvfiles[i])
    df = df.drop(['Unnamed: 0'],axis=1)
    fig = plt.figure(figsize =(12, 7))
    plt.xlabel("Algorithm")
    plt.ylabel(columnname[0])
    plt.boxplot(df, labels=df.columns)
    # plt.savefig('F:/朱老师课题/自己的代码/研究点一/deal results/picture/svc/' + columnname[0] + '.png')
    # plt.savefig('E:/360downloads/5000-2000到balance/picture/compare_methods/rf/' + columnname[0] + '.png')
    plt.show()