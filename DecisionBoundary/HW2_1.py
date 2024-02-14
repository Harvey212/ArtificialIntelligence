#!python
#!/usr/bin/env python
import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np
#-----------------------------------------------------------------------
def gradient_descent(matrix,weight,y):
	iteration=100000
	threshold=0.05
	learning_rate=0.01
	iterr=0
	while iterr<iteration:
		predict=np.matmul(matrix,weight)
		diff=np.array(y-predict)
		sign=np.sign(diff)
		gradient=(-1/1001)*np.matmul(np.transpose(matrix),sign)
		abso=np.absolute(diff)
		err=(1/1001)*(np.sum(abso))
		weight=weight-learning_rate*gradient
		iterr+=1
		if err<threshold:
			break
	return weight,err	
#----------------------------------------------------------------------

mat = sio.loadmat('data.mat')
x = np.array(mat['x'],dtype=np.float32) 
y = np.array(mat['y'],dtype=np.float32)

one=np.ones((1001,1))
square=np.square(x)
combine=np.concatenate((one,x,square),axis=1)


weight=np.random.rand(3,1)

weightF,err=gradient_descent(combine,weight,y)

print(err)

result=np.matmul(combine,weightF)

plt.plot(x,result)        

plt.plot(x,y)
plt.grid()
plt.show()

weight_coef=round(weightF[0][0],2)
weight_x=round(weightF[1][0],2)
weight_x_square=round(weightF[2][0],2)
print("Prediction curve: {} + {} x + {} x^2 =0".format(weight_coef,weight_x,weight_x_square))
