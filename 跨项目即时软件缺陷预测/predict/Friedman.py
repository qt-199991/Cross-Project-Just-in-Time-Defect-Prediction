import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from scipy.special import comb
import seaborn as sns
# from statsmodels.stats.libqsturng import psturng
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import rankdata

# Define the data for each method
# a1 = 'E:/360downloads/5000-2000到balance/meanall_measure/nb/F1_nb_.csv'
a1 = 'C:/Users/qiutian/Desktop/11/11/AUC_nb.csv'
a1 = pd.read_csv(a1)
name = ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'kwm']
# method_1 = [3, 5, 5, 5, 5]
# method_2 = [2, 3, 4, 5, 1]
# method_3 = [1, 2, 3, 5, 4]
# method_4 = [4, 5, 3, 1, 2]
# method_5 = [5, 4, 1, 2, 3]
# method_1 = [6 - item for item in method_1]
# method_2 = [6 - item for item in method_2]
# method_3 = [6 - item for item in method_3]
# method_4 = [6 - item for item in method_4]
# method_5 = [6 - item for item in method_5]
method_1 = a1[name[0]]
method_2 = a1[name[1]]
method_3 = a1[name[2]]
method_4 = a1[name[3]]
method_5 = a1[name[4]]
method_1 = [1 - item for item in method_1]
method_2 = [1 - item for item in method_2]
method_3 = [1 - item for item in method_3]
method_4 = [1 - item for item in method_4]
method_5 = [1 - item for item in method_5]
df = pd.DataFrame({"BF": method_1, "DFAC": method_2, "DMDA_JFR": method_3,
                   "HISNN": method_4, "SIKMM": method_5})
# Combine the data into a 2D array
data = np.array([method_1, method_2, method_3,method_4,method_5])
# print(data)
data = np.transpose(data)


n, k = data.shape
ranks = np.zeros((n, k))
for i in range(n):
    ranks[i,:] = rankdata(data[i,:])
avg_ranks = np.mean(ranks, axis=0)
ranks = np.transpose(ranks)
print(ranks)
# Perform the Friedman test
statistic, p_value = friedmanchisquare(*data)

# Print the results
print("Friedman test statistic: ", statistic)
print("p-value: ", p_value)

avg_ranks = np.mean(ranks, axis=1)
print(avg_ranks)

# Calculate the rank for each method
# ranks = np.apply_along_axis(rankdata, 1, -data)

# Print the ranks for each method
# for i, rank in enumerate(ranks):
#     print(f"Method {i+1} ranks: ", rank)

# Calculate the average rank for each method

# Print the average rank for each method
# for i, avg_rank in enumerate(avg_ranks):
#     print(f"Method {i+1} average rank: ", avg_rank)


df_melted = pd.melt(df, var_name="Method", value_name="Performance")

# Add the ranks to the melted data
df_melted["Rank"] = ranks.flatten()

# Define the colors for each method
colors = ["red", "green", "blue", "orange", "purple"]

# Draw the swarm plot
# sns.swarmplot(x="Method", y="Rank", hue="Method", palette=colors, data=df_melted, size=8)
sns.swarmplot(x="Method", y="Rank", palette=colors, data=df_melted, size=8)

# Add title and axis labels
plt.title("Swarm Plot of Ranking")
plt.xlabel("Method")
plt.ylabel("Rank")

# Draw horizontal lines for each rank
for i in range(1, len(df.columns)+1):
    plt.hlines(i, xmin=-0.4, xmax=len(df.columns)-0.6, colors="gray", linestyles="dashed")

# Show the plot
plt.show()


# # Calculate the critical value for the Nemenyi's post-hoc test
# num_methods = data.shape[0]
# num_datasets = data.shape[1]
# q = psturng(0.05 / (2 * comb(num_methods, 2)), num_methods, np.inf)
# cd = q * np.sqrt(num_methods * (num_methods + 1) / (6 * num_datasets))


# Perform the Nemenyi's post-hoc test
# for i in range(num_methods):
#     for j in range(i + 1, num_methods):
#         diff = avg_ranks[i] - avg_ranks[j]
#         if np.abs(diff) > cd:
#             print(f"Methods {i+1} and {j+1} are significantly different (difference = {diff}, CD = {cd}).")


