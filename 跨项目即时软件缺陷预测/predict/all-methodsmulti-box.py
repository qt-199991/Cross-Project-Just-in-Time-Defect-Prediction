# import matplotlib.pyplot as plt
# import pandas as pd
#
# resultspath = 'E:/360downloads/5000-2000到balance/meanall_measure/lr/'
# name = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'kwm']
# mea = ['F1_lr_', 'Balance_lr_', 'MCC_lr_', 'G_Mean_lr_', 'Recall_lr_']
# df0 =  pd.read_csv(resultspath + mea[0] + '.csv')
# df1 =  pd.read_csv(resultspath + mea[1] + '.csv')
# df2 =  pd.read_csv(resultspath + mea[2] + '.csv')
# df3 =  pd.read_csv(resultspath + mea[3] + '.csv')
# df4 =  pd.read_csv(resultspath + mea[4] + '.csv')
# # print(df0[name[0]])
# # data是acc中三个箱型图的参数
# data = [df0[name[0]],df0[name[1]],df0[name[2]],df0[name[3]], df0[name[4]]]
# # data2 是F1 score中三个箱型图的参数
# data2 = [df1[name[0]],df1[name[1]],df1[name[2]],df1[name[3]], df1[name[4]]]
# # data3 是IoU中三个箱型图的参数
# data3 = [df2[name[0]],df2[name[1]],df2[name[2]],df2[name[3]], df2[name[4]]]
#
# data4 = [df3[name[0]], df3[name[1]], df3[name[2]], df3[name[3]], df3[name[4]]]
#
# data5 = [df4[name[0]], df4[name[1]], df4[name[2]], df4[name[3]], df4[name[4]]]
# # 箱型图名称
# labels = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'ISKMM']
# # 五个箱型图的颜色 RGB （均为0~1的数据）
# colors = [(176 / 255., 224 / 255., 230 / 255.), (220 / 255., 220 / 255., 220 / 255.),
#           (255 / 255., 235 / 255., 205 / 255.), (188 / 255., 143 / 255., 143 / 255.),
#           (255 / 255., 0 / 255., 0 / 255.)]
# # 绘制箱型图
#
# plt.figure(figsize=(18,10))
# # patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
# bplot = plt.boxplot(data, patch_artist=True, labels=labels, positions=(1, 1.6, 2.2, 2.8, 3.4), widths=0.6,showfliers = False)
# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)
#
# bplot2 = plt.boxplot(data2, patch_artist=True, labels=labels, positions=(4.4, 5.0, 5.6, 6.2, 6.8), widths=0.6,showfliers = False)
# for patch, color in zip(bplot2['boxes'], colors):
#     patch.set_facecolor(color)
#
# bplot3 = plt.boxplot(data3, patch_artist=True, labels=labels, positions=(7.8, 8.4, 9.0, 9.6, 10.2), widths=0.6,showfliers = False)
# for patch, color in zip(bplot3['boxes'], colors):
#     patch.set_facecolor(color)
#
# bplot4 = plt.boxplot(data4, patch_artist=True, labels=labels, positions=(11.2, 11.8, 12.4, 13.0, 13.6), widths=0.6,showfliers = False)
# for patch, color in zip(bplot4['boxes'], colors):
#     patch.set_facecolor(color)
#
# bplot5 = plt.boxplot(data5, patch_artist=True, labels=labels, positions=(14.6, 15.2, 15.8, 18.4, 17.0), widths=0.6,showfliers = False)
# for patch, color in zip(bplot5['boxes'], colors):
#     patch.set_facecolor(color)
#
# x_position = [1, 4.4, 7.8, 11.2, 14.6]
# x_position_fmt = ["F1-measure", "Balance", "MCC", 'G-Mean', 'Recall']
#
# plt.xticks([i + 2.4 / 2 for i in x_position], x_position_fmt, fontsize=18)
# plt.yticks(fontsize=18)
# plt.ylabel('lr(%)', fontsize=18)
# plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
#
#
# plt.legend(bplot['boxes'], labels, bbox_to_anchor=(0.5, -0.13), loc = 8, ncol = 10, fontsize=18)
# plt.savefig('E:/360downloads/5000-2000到balance/picture/compare_methods/lr/' + 'lr' + '.png', bbox_inches='tight')
# # plt.show()
#
#



import matplotlib.pyplot as plt
import pandas as pd

resultspath = 'E:/朱老师课题/自己的代码/研究点一/1/5000-2000tobalance/other_data/meanall_measure/svc/'
name = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'kwm']
mea = ['F1_svc_', 'Balance_svc_', 'AUC_svc_', 'G_Mean_svc_']
df0 =  pd.read_csv(resultspath + mea[0] + '.csv')
df1 =  pd.read_csv(resultspath + mea[1] + '.csv')
df2 =  pd.read_csv(resultspath + mea[2] + '.csv')
df3 =  pd.read_csv(resultspath + mea[3] + '.csv')
# print(df0[name[0]])
# data是acc中三个箱型图的参数
data = [df0[name[0]],df0[name[1]],df0[name[2]],df0[name[3]], df0[name[4]]]
# data2 是F1 score中三个箱型图的参数
data2 = [df1[name[0]],df1[name[1]],df1[name[2]],df1[name[3]], df1[name[4]]]
# data3 是IoU中三个箱型图的参数
data3 = [df2[name[0]],df2[name[1]],df2[name[2]],df2[name[3]], df2[name[4]]]

data4 = [df3[name[0]], df3[name[1]], df3[name[2]], df3[name[3]], df3[name[4]]]

# 箱型图名称
labels = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'ISKMM']
# 五个箱型图的颜色 RGB （均为0~1的数据）
colors = [(176 / 255., 224 / 255., 230 / 255.), (220 / 255., 220 / 255., 220 / 255.),
          (255 / 255., 235 / 255., 205 / 255.), (188 / 255., 143 / 255., 143 / 255.),
          (255 / 255., 0 / 255., 0 / 255.)]
# 绘制箱型图

plt.figure(figsize=(14,9))
# patch_artist=True-->箱型可以更换颜色，positions=(1,1.4,1.8)-->将同一组的三个箱间隔设置为0.4，widths=0.3-->每个箱宽度为0.3
bplot = plt.boxplot(data, patch_artist=True, labels=labels, positions=(1, 1.6, 2.2, 2.8, 3.4), widths=0.6,showfliers = False)
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

bplot2 = plt.boxplot(data2, patch_artist=True, labels=labels, positions=(4.4, 5.0, 5.6, 6.2, 6.8), widths=0.6,showfliers = False)
for patch, color in zip(bplot2['boxes'], colors):
    patch.set_facecolor(color)

bplot3 = plt.boxplot(data3, patch_artist=True, labels=labels, positions=(7.8, 8.4, 9.0, 9.6, 10.2), widths=0.6,showfliers = False)
for patch, color in zip(bplot3['boxes'], colors):
    patch.set_facecolor(color)

bplot4 = plt.boxplot(data4, patch_artist=True, labels=labels, positions=(11.2, 11.8, 12.4, 13.0, 13.6), widths=0.6,showfliers = False)
for patch, color in zip(bplot4['boxes'], colors):
    patch.set_facecolor(color)

x_position = [1, 4.4, 7.8, 11.2]
x_position_fmt = ["F1-measure", "Balance", "AUC", 'G-Mean']

plt.xticks([i + 2.4 / 2 for i in x_position], x_position_fmt, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('SVC(%)', fontsize=18)
plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3


plt.legend(bplot['boxes'], labels, bbox_to_anchor=(0.485, -0.14), loc = 8, ncol = 10, fontsize=18)
plt.savefig('C:/Users/qiutian/Desktop/photo/bar/' + 'svc' + '.png', bbox_inches='tight')
# plt.show()







