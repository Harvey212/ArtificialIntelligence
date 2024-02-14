import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv

df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW3\introML2019F.task3.train.csv',engine='python')

#print(df.keys())
X_train=df.iloc[:, 0:26]
Y_train=df.iloc[:, 26]

test=pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW3\introML2019F.task3.test.shuffled.noanswers.csv')
#print(test.keys())
X_test=test.iloc[:, 0:26]
selector=SelectKBest(score_func=chi2,k=25) 

X_clf_new=selector.fit_transform(X_train,Y_train)
clf = RandomForestClassifier(criterion='entropy').fit(X_clf_new,Y_train)

X_test_clean = selector.transform(X_test)
predict=clf.predict(X_test_clean)
predict.reshape(-1,1)

data =pd.DataFrame(predict,columns=['Category']) 
data.index = np.arange(1, len(data)+1)
data.to_csv('HW3.csv',index = True,index_label='ID') 
