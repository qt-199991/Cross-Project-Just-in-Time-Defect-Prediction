# import Orange
# # 查看函数用法
# # help(Orange.evaluation.graph_ranks)
# import matplotlib.pyplot as plt
# names = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'iskmm']
# avranks =  [3.13333333, 3.85, 1.63333333,  3.5, 2.88333333]
# cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 dataset
# colors = {'BF': 'r', 'DFAC': 'g', 'DMDA_JFR': 'b', 'HISNN': 'y', 'iskmm': 'm'}
# Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5, filename="E:/360downloads/5000-2000到balance/picture/compare_methods/cd.png")
# plt.show()
import Orange
import matplotlib.pyplot as plt
names = ["first", "third", "second", "fourth" ]
avranks =  [1.9, 3.2, 2.8, 3.3 ]
cd = Orange.evaluation.compute_CD(avranks, 30) #tested on 30 datasets
Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()
