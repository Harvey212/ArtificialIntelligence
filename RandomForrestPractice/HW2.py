import pandas as pd 
import numpy as np
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
#from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import csv

df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW2\introML2019F.task2.train.csv',engine='python')

#with open('introML2019F.task2.train.csv', 'r') as csvfile:
#	csvreader = csv.reader(csvfile)
    
#print(df.keys())
X_train=df.iloc[:, 0:21]
Y_train=df.iloc[:, 21]

X=np.array(X_train)
Y=np.array(Y_train)
clf = RandomForestClassifier(criterion='entropy').fit(X,Y)
#logreg = LogisticRegression(multi_class='multinomial',solver='newton-cg',penalty='l2',max_iter=1000).fit(X, Y)

#print(np.std(X,axis=0))

test=pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW2\introML2019F.task2.test.shuffled.noanswers.csv')
#print(test.keys())
X_test=test.iloc[:, 0:21]
predict=clf.predict(X_test)
predict.reshape(-1,1)

data =pd.DataFrame(predict,columns=['Category'])
data.index = np.arange(1, len(data)+1)
data.to_csv('HW2.csv',index = True,index_label='ID')
