from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
import numpy as np
# read in data
#%%
# x=pd.read_excel('./data/全部整理数据.xlsx')
test=pd.read_csv("./data/test_new.csv",header=0)
train=pd.read_csv("./data/train_new.csv",header=0)
# col=['电影ID','用户ID	','评分'	,'时间戳','用户ID','性别',	'年龄','职业'	,'邮编','电影ID','电影名',	'上市年份	','Animation','Adventure','Comedy','Action','Drama','Thriller','Crime','Romance','Children','Documentary','Sci-Fi','Horror','Western','Mystery','Film-Noir','War','Fantasy','Musical']

#%%
print (test.columns)
#%%

train_Y=train['评分']
train_X=train.copy()
test_Y=test['评分']
test_X=test.copy()

train_X.drop([u'评分',u'电影名',u'电影ID2',u'用户ID2'],axis=1,	inplace=True)
# train_X.drop([0,1,2,4,11,12],axis=1)
test_X.drop([u'评分',u'电影名',u'电影ID2',u'用户ID2'],axis='columns',	inplace=True)
print (test.columns)
print (train_X.columns)

#%%
train_X.astype('float')
train_Y.astype('float')
test_X.astype('float')
test_Y.astype('float')
#%%数据转换
train_XX=np.array(train_X)
train_YY=np.array(train_Y)
test_YY=np.array(test_Y)
test_XX=np.array(test_X)

#%%
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')#读取训练数据
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')#读取测试数据
# specify parameters via map通过映射指定参数
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 2

# bst = xgb.train(param, dtrain, num_round)#用XGBoost训练模型
# make prediction#用测试数据做验证
# preds = bst.predict(dtest)

model = XGBClassifier()
model.fit(train_X, train_Y)