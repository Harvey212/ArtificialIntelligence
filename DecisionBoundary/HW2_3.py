#!python
#!/usr/bin/env python
import scipy.io as sio
import numpy as np
#-----------------------------------------------------------------------
def gradient_descent(matrix,weight,y):
	iteration=100000
	threshold=0.05
	learning_rate=0.01
	iterr=0
	row=matrix.shape[0]
	while iterr<iteration:
		z=np.matmul(matrix,weight)
		prob=1/(1+np.exp(-z))

		MLE_log_prob=np.matmul(np.transpose(y),np.log(prob+1e-10))+np.matmul(np.transpose((1-y)),np.log(1-prob+1e-10))
		cost=(1/row)*abs(MLE_log_prob)

		gradient=(1/row)*np.matmul(np.transpose(matrix),(prob-y))

		weight=weight-learning_rate*gradient
		iterr+=1
		if cost<threshold:
			break
	return weight,cost
#-----------------------------------------------------------------------
mat = sio.loadmat('train.mat')
train_x1 = np.array(mat['x1'],dtype=np.float32)
train_x2 = np.array(mat['x2'],dtype=np.float32)
coef_train=np.ones((train_x1.shape[0],1))
train_y = np.array(mat['y'],dtype=np.float32)


train_X=np.concatenate((coef_train,train_x1,train_x2),axis=1)

weight=np.random.rand(3,1)

weightF,cost=gradient_descent(train_X,weight,train_y)

weight_coef=round(weightF[0][0],2)
weight_x1=round(weightF[1][0],2)
weight_x2=round(weightF[2][0],2)

#print(weightF)
#print(cost)

test = sio.loadmat('test.mat')
test_x1 = np.array(test['x1'],dtype=np.float32)
test_x2 = np.array(test['x2'],dtype=np.float32)
test_y = np.array(test['y'],dtype=np.float32)
coef_test=np.ones((test_x1.shape[0],1))
test_X=np.concatenate((coef_test,test_x1,test_x2),axis=1)

Z_test=np.matmul(test_X,weightF)
prob=1/(1+np.exp(-Z_test))

predict=np.zeros((len(test_y),1))

for i in range(0,len(test_y)):
	if prob[i][0]>0.5:
		predict[i][0]=1

diff=abs(predict-test_y)

TestError=(100/len(test_y))*(np.sum(diff))

accuracy=100-TestError

print("Decision boundary: {} x1 + {} x2 + {} =0".format(weight_x1,weight_x2,weight_coef))
print("Accuracy: {0:.2f}%".format(accuracy))
print("Test Error: {0:.2f}%".format(TestError))
