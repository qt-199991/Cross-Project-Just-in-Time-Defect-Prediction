#这里是所有方法迁移的平均值而不是最大值
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import  pandas as pd

plt.rcParams['font.sans-serif']=['SimHei'] # 解决中文乱码
intra_cross = 'E:/360downloads/5000-2000到balance/inter project/intra_svc.csv'
df1 = pd.read_csv(intra_cross)

labels = ['broadleaf','go','nova', 'openstack','platform','qt']
a1 = 'E:/360downloads/5000-2000到balance/compare_iskmm_inter/mean-allmethods/svc/cross_project_BF.csv'
a1 = pd.read_csv(a1)
b1 = 'E:/360downloads/5000-2000到balance/compare_iskmm_inter/mean-allmethods/svc/cross_project_DFAC.csv'
b1 = pd.read_csv(b1)
c1= 'E:/360downloads/5000-2000到balance/compare_iskmm_inter/mean-allmethods/svc/cross_project_DMDA_JFR.csv'
c1 = pd.read_csv(c1)
d1 = 'E:/360downloads/5000-2000到balance/compare_iskmm_inter/mean-allmethods/svc/cross_project_HISNN.csv'
d1 = pd.read_csv(d1)
e1 = 'E:/360downloads/5000-2000到balance/compare_iskmm_inter/mean-allmethods/svc/cross_project_ISKMM.csv'
e1 = pd.read_csv(e1)
name_column = ['AUC', 'F1', 'MCC', 'G-Mean', 'Balance', 'Recall', 'PF', 'Precision']
for name in name_column:
    a = a1[name]
    b = b1[name]
    c = c1[name]
    d = d1[name]
    e = e1[name]
    inter = df1[name]

    x = np.arange(len(labels))  # 标签位置
    width = 0.1  # 柱状图的宽度，可以根据自己的需求和审美来改

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width*2, a, width, label='BF', color = (255 / 255., 125 / 255., 64 / 255.))
    rects2 = ax.bar(x - width+0.01, b, width, label='DFAC', color = (188 / 255., 143 / 255., 143 / 255.))
    rects3 = ax.bar(x + 0.02, c, width, label='DMDA-JFR', color = (160 / 255., 102 / 255., 211 / 255.))
    rects4 = ax.bar(x + width + 0.03, d, width, label='HISNN', color = (221 / 255., 160 / 255., 221 / 255.))
    rects5 = ax.bar(x + width * 2 + 0.04, e, width, label='ISKMM', color = (255 / 255., 99 / 255., 71 / 255.))
    rects6 = ax.bar(x + width * 3 + 0.05, inter, width, label='WPDP', color = (255 / 255., 227 / 255., 132 / 255.))


    # 为y轴、标题和x轴等添加一些文本。
    ax.set_ylabel(name, fontsize=16)
    # ax.set_xlabel('pro', fontsize=16)
    # ax.set_title('这里是标题')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(0.5, -0.2), loc = 8, ncol = 10)
    # plt.legend(bplot['boxes'], labels, bbox_to_anchor=(0.5, -0.1), loc = 8, ncol = 10)

    # def autolabel(rects):
    #     """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3点垂直偏移
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')

    # autolabel(rects1)
    # autolabel(rects2)
    # autolabel(rects3)
    # autolabel(rects4)
    # autolabel(rects5)
    # autolabel(rects6)

    fig.tight_layout()

    # plt.show()
    plt.savefig('E:/360downloads/5000-2000到balance/picture/inter_cross/multi-methods/svc/intra_cross_' + name + '.png',bbox_inches='tight')


