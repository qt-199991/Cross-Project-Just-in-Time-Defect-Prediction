# import matplotlib.pyplot as plt
# import pandas as pd
# from mpl_toolkits import mplot3d
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# dfi = pd.read_csv('E:/360downloads/parameters/parameters_results.csv')
# x = dfi.iloc[:, 9:10]
# print(x)
# y = dfi.iloc[:, 10:]
# z = dfi.iloc[:, 2:3]
#
# ax.scatter(x, y, z)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

dfi = pd.read_csv('E:/360downloads/parameters/parameters_results.csv')
x = dfi.iloc[:, 1:2]
y = dfi.iloc[:, 2:3]
z = dfi.iloc[:, 4:5]

# 创建网格
X, Y = np.meshgrid(x, y)

# 创建3D图
fig = plt.figure()
ax = fig.gca(projection='3d')

# 绘制3D colormap图
surf = ax.plot_surface(X, Y, z, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)

# 设置坐标轴标签和标题
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Colormap Example')

# 显示图像
plt.show()
