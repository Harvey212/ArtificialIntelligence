#!python
#!/usr/bin/env python
import scipy.io as sio
import numpy as np
from sklearn.linear_model import LogisticRegression 

#-----------------------------------------------------------------------

mat = sio.loadmat('train.mat')
train_x1 = np.array(mat['x1'],dtype=np.float32)
train_x2 = np.array(mat['x2'],dtype=np.float32)
coef_train=np.ones((train_x1.shape[0],1))
train_y = np.array(mat['y'],dtype=np.float32)

train_X=np.concatenate((train_x1,train_x2,coef_train),axis=1)
fit_Train = LogisticRegression(random_state=0, solver='liblinear').fit(train_X, np.ravel(train_y))
weight=fit_Train.coef_
weight_x1=round(weight[0][0],2)
weight_x2=round(weight[0][1],2)
weight_coef=round(weight[0][2],2)

test = sio.loadmat('test.mat')
test_x1 = np.array(test['x1'],dtype=np.float32)
test_x2 = np.array(test['x2'],dtype=np.float32)
test_y = np.array(test['y'],dtype=np.float32)
coef_test=np.ones((test_x1.shape[0],1))
test_X=np.concatenate((test_x1,test_x2,coef_test),axis=1)

accuracy=(100)*fit_Train.score(test_X,test_y)
TestError=100-accuracy

print("Decision boundary: {} x1 + {} x2 + {} =0".format(weight_x1,weight_x2,weight_coef))
print("Accuracy: {0:.2f}%".format(accuracy))
print("Test Error: {0:.2f}%".format(TestError))
