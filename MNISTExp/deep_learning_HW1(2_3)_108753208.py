import scipy.io as sio
import matplotlib.pyplot as plt 
import numpy as np
#-----------------------------------------------------------------------
def gradient_descent(x,y):
	a=0                        #square term
	m=0                        #there is only one feature for x, so we only need a constant for 1 dimension
	b=0                        #y'=ax^2+mx+b
	iteration=10000
	n=len(x)
	learning_rate=0.01

	for i in range(iteration):
		y_predict=a*(x**2)+m*x+b                                      
		error=np.sqrt((1/n)*np.sum(np.square(y-y_predict)))
		ad=(-2/n)*(np.matmul(np.transpose(x**2),(y-y_predict)))       #a derivitive
		md=(-2/n)*(np.matmul(np.transpose(x),(y-y_predict)))          #m derivitive  
		bd=(-2/n)*np.sum(y-y_predict)                                 #b derivitive
		a=a-learning_rate*ad
		m=m-learning_rate*md
		b=b-learning_rate*bd

	return a,m,b,error	
#----------------------------------------------------------------------

mat = sio.loadmat('data.mat')
x = np.array(mat['x'],dtype=np.float32) 
y = np.array(mat['y'],dtype=np.float32)
a,m,b,error=gradient_descent(x,y)
y_p=a*(x**2)+m*x+b
print(f'error: {error},a:{a}, m:{m}, b:{b}')
#plot
plt.plot(x,y_p)        
plt.plot(x,y)
plt.grid()
plt.show()