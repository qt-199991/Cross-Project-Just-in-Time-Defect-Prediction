import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
np.set_printoptions(threshold=np.inf)

for name in ['BF', 'DFAC', 'DMDA_JFR', 'HISNN', 'ISKMM']:
    # path2 = 'F:/朱老师课题/自己的代码/研究点一/svc results/' +name + '.csv'
    path1 = 'E:/360downloads/5000-2000到balance/lr results/' + name + '.csv'
    df = pd.read_csv(path1)
    newdf = pd.DataFrame(np.repeat(df.values,6,axis=0))
    newdf.columns = df.columns
    newdf.to_csv('E:/360downloads/5000-2000到balance/lr/' + name + '1.csv')