import pandas as pd 
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
import csv

df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW1\introML2019F.task1.train.shuffled.csv')

#print(df.keys())

X_train=df.iloc[:, 0:21]
Y_train=df.iloc[:, 21]

#print(X_train.shape)
#print(is_numeric_dtype(df['feature_15']))
clf = RandomForestClassifier(criterion='entropy').fit(X_train,Y_train)
#tree.plot_tree(clf.fit(X,Y))

test=pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW1\introML2019F.task1.test.shuffled.noanswers.csv')
#print(test.keys())
X_test=test.iloc[:, 0:21]
predict=clf.predict(X_test)
predict.reshape(-1,1)

data =pd.DataFrame(predict,columns=['Category']) 
data.index = np.arange(1, len(data)+1)
data.to_csv('HW1.csv',index = True,index_label='ID') 

#df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW1\HW1.csv')
#print(df.keys)