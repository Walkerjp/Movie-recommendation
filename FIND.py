from sklearn import tree
import pandas as pd
import numpy as np
import csv
#%%
TJ_itemcf=pd.read_csv("./data/itemcf_movie2无兴趣度.csv",header=0)
TJ_lfm=pd.read_csv("./data/lfm_movie无兴趣度.csv",header=0)
TJ_itemcf.drop(1)
Use=pd.read_table("./data/客户整理.xlsx",header=0,)
Movie=pd.read_table("./data/电影整理.xlsx",header=0)
# './data/ratings.dat'
#%%




#%%
TJ_itemcf.astype('float')

#%%
# 数据框生成，
# TJ_itemcf.index=1
# TJ_itemcf.index

TesT=pd.DataFrame([])#columns=X.columns
ii=0
for i in range(1,6040+1,1):
    for j in range(1,10,1):
        I=Use.iloc[i]
        jj=TJ_itemcf.iloc[i,j]
        J=Movie.iloc[j]
        IJ=pd.merge(I, J)
        TesT.loc[ii] = IJ
        ii += 1
print('运行无误')
