# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../data"))

# Any results you write to the current directory are saved as output.

b1 = pd.read_csv('../data/submission_kernel02.csv').rename(columns={'TARGET':'dp1'})
b2 = pd.read_csv('../data/submission_xgboost.csv').rename(columns={'TARGET':'dp2'})
b3 = pd.read_csv('../data/WEIGHT_AVERAGE_RANK.csv').rename(columns={'TARGET':'dp3'})
b1 = pd.merge(b1,b2,how='left', on='SK_ID_CURR')
b1 = pd.merge(b1,b3,how='left', on='SK_ID_CURR')

# 三个模型的单独得分别为 0.798 0.798 0.
# 此处做加权平均

b1['TARGET'] = (b1['dp1'] * 0.3) + (b1['dp2'] * 0.2) + (b1['dp3'] * 0.5)
b1[['SK_ID_CURR','TARGET']].to_csv('Submission_data_blend2.csv', index=False)
print("__________finish____________")


