import pandas as pd 
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, f_regression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import csv
import numpy.ma as ma
from sklearn.impute import SimpleImputer

df = pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW4\introML2019F.task4.train.csv',engine='python')

#print(df.iloc[0,:])
X_train=df.iloc[:, 0:26]
Y_train=df.iloc[:, 26]
##########################
#a=np.array([0,1,2,3])
#a = ma.masked_where(a == 2, a)
#mm=np.nanmean(a)
#print(a)
#print(mm)
##########################
X_ex=ma.masked_where(X_train==-1,X_train)
imp= SimpleImputer(missing_values=np.nan, strategy='most_frequent',add_indicator=True)
X_new=imp.fit_transform(X_ex,Y_train)

#clf = tree.DecisionTreeClassifier(criterion='entropy').fit(X_new,Y_train)

######################################
#X=np.array(X_train)
#X_ex_median=np.nanmedian(X_ex,axis=0)
#print(X_ex_mean[0])
#for i in range(np.size(X,1)):
#	np.where(X[:,i]==-1, X_ex_median[i], X[:,i]) 
###############################################

test=pd.read_csv(r'C:\Users\chrystal212\Desktop\ML_HW4\introML2019F.task4.test.shuffled.noanswers.csv')
#print(test.keys())
X_test=test.iloc[:, 0:26]
X_ex_test=ma.masked_where(X_test==-1,X_test)
X_new_test=imp.transform(X_ex_test)

#################################################
#XX=np.array(X_test)
#X_ex_median_test=np.nanmedian(X_ex_test,axis=0)
#for i in range(np.size(XX,1)):
#	np.where(XX[:,i]==-1, X_ex_median_test[i], XX[:,i])
####################################################

selector=SelectKBest(k=25) 
X_clf_new=selector.fit_transform(X_new,Y_train)
clf = RandomForestClassifier(criterion='entropy').fit(X_clf_new,Y_train)

X_test_clean = selector.transform(X_new_test)


predict=clf.predict(X_test_clean)
predict.reshape(-1,1)

data =pd.DataFrame(predict,columns=['Category']) 
data.index = np.arange(1, len(data)+1)
data.to_csv('HW4.csv',index = True,index_label='ID') 
