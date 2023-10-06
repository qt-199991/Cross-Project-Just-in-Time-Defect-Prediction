# import Orange
# import matplotlib.pyplot as plt
#
# names = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'iskmm']
# avranks = [3.13333333, 3.85, 1.63333333, 3.5, 2.88333333]
#
# cd = Orange.evaluation.compute_CD(avranks, 30)
#
# fig, ax = plt.subplots()
# ax.plot(avranks, names, 'o')
# for i, name in enumerate(names):
#     ax.plot([avranks[i] - cd, avranks[i] + cd], [name, name], '-')
#
# ax.set_xlabel('Average rank')
# ax.set_yticklabels(names)
# plt.show()
import Orange
import matplotlib.pyplot as plt
names = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'iskmm']
avranks = [3.13333333, 3.85, 1.63333333, 3.5, 2.88333333]

cd = Orange.evaluation.compute_CD(avranks, 30)

fig, ax = plt.subplots()
for i in range(len(names)):
    ax.plot([avranks[i] - cd, avranks[i] + cd], [i, i], 'k-', lw=1)
    ax.plot([avranks[i]], [i], 'o', markersize=8, color='white', mec='k', mew=1)

ax.set_xlabel('Average rank')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.invert_yaxis()

plt.show()
