import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

waters = ('Broadleaf','Go','Nova', 'Openstack','Platform','QT')
crosspath = 'F:/朱老师课题/自己的代码/研究点一/1/5000-2000到balance/compare_iskmm_inter/max/nb/cross_project_ISKMM.csv'
intra_cross = 'F:/朱老师课题/自己的代码/研究点一/1/5000-2000到balance/inter project/intra_nb.csv'
df = pd.read_csv(crosspath)
df1 = pd.read_csv(intra_cross)
name_column = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision']
for name in name_column:
# name = 'F1'
    cross = []
    data = df[name]
    for i in range(len(data)):
        cross.append(data[i])
    print(cross)

    intra = []
    data1 = df1[name]
    for i in range(len(data1)):
        intra.append(data1[i])
    # print(intra)
    bar_width = 0.1 # 条形宽度
    index_male = np.arange(len(waters)) * 0.35
    # print('index_male', index_male)
    index_female = index_male + bar_width
    # print('index_female', index_female)
    color = [(190 / 255., 184 / 255., 220 / 255.), (250 / 255., 127 / 255., 111 / 255.)]
    labels = ['within-project', 'ISKMM']

    patches = [mpatches.Patch(color=color[i], label="{:s}".format(labels[i])) for i in range(len(color))]
    # 使用两次 bar 函数画出两组条形图
    # plt.bar(index_male, height=intra, width=bar_width, color='mediumaquamarine', label='intra_project')
    # plt.bar(index_female, height=cross, width=bar_width, color='salmon', label='cross_project')
    # plt.title('项目内与跨项目' + name + '对比')  # 图形标题
    plt.bar(index_male, height=intra, width=bar_width, color = color[0])
    plt.bar(index_female, height=cross, width=bar_width, color = color[1])
    plt.legend()  # 显示图例
    plt.xticks(index_male + bar_width / 2, waters, fontsize=14, rotation=340)
    plt.yticks(fontsize=16)
    if name == 'F1':
        name = 'F1-measure'
    # print(name)
    plt.ylabel(name + '(%)' + '/NB', fontsize=14)  # 纵坐标轴标题
    ax = plt.gca()
    ax.legend(handles=patches, bbox_to_anchor=(0.83, -0.12), ncol=3, fontsize=14)
    # plt.savefig('F:/朱老师课题/自己的代码/研究点一/deal results/picture/intra_cross_' + name + '.png')
    plt.savefig('F:/朱老师课题/自己的代码/研究点一/1/5000-2000到balance/picture/inter_cross/iskmm-inter/nb/nb-intra-cross-' + name + '.png',bbox_inches='tight')
    plt.clf() #不添加这个所有的表格都一样，都被第一个刷新了