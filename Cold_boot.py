from sklearn import tree
import pandas as pd
import numpy as np
import csv
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
#%%
print (test.columns)
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
#%%把训练集用于训练参数
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_XX, train_YY)
#%%



k=clf.predict([test_XX])
jj=clf.predict_proba([test_XX])

#%%

#%%
print(jj)
print (k)
#把测试集合放进去




#%%
from sklearn import tree
from sklearn import metrics
import pandas as pd

x_train=pd.read_csv('F:\\重要书籍\\数据分析\\train.csv')
X=x_train.copy().drop(columns = ['评分'])
Y=x_train['评分']
x_test=pd.read_csv('F:\\重要书籍\\数据分析\\test.csv')
X_test=x_text.copy().drop(columns = ['评分'])
Y_test=x_test['评分']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
#%%
y_testpridict=clf.predict(X_test)
accuracy=metrics.accuracy_score(Y_test,y_testpridict)
AUC=metrics.ROC_auc_score(Y_test,y_testpridict)