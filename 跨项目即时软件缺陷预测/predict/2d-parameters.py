# # kwm
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# dfi = pd.read_csv('C:/Users/qiutian/Desktop/研究点一/1/svc_parameters/kwm/parameters-results1.csv')
# x = dfi.iloc[:, 9:10]
# # x = dfi.iloc[:, 10:11]
# k1 = dfi.iloc[:, 2:3]
# k2 = dfi.iloc[:, 1:2]
# k3 = dfi.iloc[:, 4:5]
# k4 = dfi.iloc[:, 5:6]
# # k5 = dfi.iloc[:, 6:7]
# labels = ['F1-measure', 'AUC', 'G-mean', 'Balance']
# plt.plot(x,k1,'s-',color = 'brown',label="F1-measure")
# plt.plot(x,k2,'o-',color = 'green',label="AUC")
# plt.plot(x,k3,'*-',color = 'blue',label="G-mean")
# plt.plot(x,k4,'P-',color = 'red',label="Balance")
# # plt.plot(x,k5,'X-',color = 'deepskyblue',label="Recall")
# plt.yticks(fontsize=16)
# # new_ticks = ['','0','0.1Sn','0.2Sn','0.3Sn','0.4Sn','0.5Sn','0.6Sn']
# # plt.gca().set_xticklabels(new_ticks)
# plt.xticks(fontsize = 16)
# plt.xlabel("Resampling size(P)", fontsize=16)
# plt.legend(labels, bbox_to_anchor=(0.5, 1.00), loc = 8, ncol = 10, fontsize=14)
# plt.savefig('C:/Users/qiutian/Desktop/photo/parameters/' + 'parameters-kwm.png',bbox_inches='tight')
# bsw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
dfi = pd.read_csv('C:/Users/qiutian/Desktop/研究点一/1/svc_parameters/bsw/parameters-results1.csv')
# x = dfi.iloc[:, 9:10]
x = dfi.iloc[:, 10:11]
k1 = dfi.iloc[:, 2:3]
k2 = dfi.iloc[:, 1:2]
k3 = dfi.iloc[:, 4:5]
k4 = dfi.iloc[:, 5:6]
# k5 = dfi.iloc[:, 6:7]
labels = ['F1-measure', 'AUC', 'G-mean', 'Balance']
plt.plot(x,k1,'s-',color = 'brown',label="F1-measure")
plt.plot(x,k2,'o-',color = 'green',label="AUC")
plt.plot(x,k3,'*-',color = 'blue',label="G-mean")
plt.plot(x,k4,'P-',color = 'red',label="Balance")
# plt.plot(x,k5,'X-',color = 'deepskyblue',label="Recall")
plt.yticks(fontsize=16)
new_ticks = ['','0','0.1Sn','0.2Sn','0.3Sn','0.4Sn','0.5Sn','0.6Sn']
plt.gca().set_xticklabels(new_ticks)
plt.xticks(fontsize = 16)
plt.xlabel("Resampling size(P)", fontsize=16)
plt.legend(labels, bbox_to_anchor=(0.5, 1.00), loc = 8, ncol = 10, fontsize=14)
plt.savefig('C:/Users/qiutian/Desktop/photo/parameters/' + 'parameters-bsw.png',bbox_inches='tight')

