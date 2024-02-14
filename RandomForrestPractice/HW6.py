import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy.ma as ma
from sklearn.impute import SimpleImputer
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW6\introML2019F.task6.train.csv',engine='python')
#print(df.iloc[0,:])

mymap = {'A':1, 'B':2, 'C':3, 'D':4}
df=df.applymap(lambda s: mymap.get(s) if s in mymap else s)

test=pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW6\introML2019F.task6.test.shuffled.noanswers.csv')
#print(test.iloc[0,:])
#print(test.keys())
test=test.applymap(lambda s: mymap.get(s) if s in mymap else s)

X_train_NU=df.iloc[:, 0:21]
X_train_NO=df.iloc[:, 21]
X_train_NO=np.array(X_train_NO).reshape(-1,1)
X_train=np.concatenate((X_train_NU,X_train_NO),axis=1)
Y_train=df.iloc[:, 22]

X_ex=ma.masked_where(X_train==-1,X_train)
imp= SimpleImputer(missing_values=np.nan, strategy='most_frequent',add_indicator=True)
X_new=imp.fit_transform(X_ex,Y_train)
#print(np.array(X_train_NO)[0:5])
X_test_NU=test.iloc[:, 0:21]
X_test_NO=test.iloc[:, 21]
X_test_NO=np.array(X_test_NO).reshape(-1,1)
X_test=np.concatenate((X_test_NU,X_test_NO),axis=1)
X_ex_test=ma.masked_where(X_test==-1,X_test)
X_new_test=imp.transform(X_ex_test)


########################################
#X_train_NO_T,alphabet=pd.factorize(X_train_NO)
#X_test_NO_T,alphabet_test=pd.factorize(X_test_NO)
#print(alphabet)
#print(df.dtypes)
#print(test.dtypes)
##########################
selector=SelectKBest(k=21) 
X_clf_new=selector.fit_transform(X_new,Y_train)
clf = RandomForestClassifier(criterion='entropy').fit(X_clf_new,Y_train)

X_test_clean = selector.transform(X_new_test)

predict=clf.predict(X_test_clean)
predict.reshape(-1,1)

data =pd.DataFrame(predict,columns=['Category']) 
data.index = np.arange(1, len(data)+1)
data.to_csv('HW6.csv',index = True,index_label='ID') 
