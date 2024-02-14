#!python
#!/usr/bin/env python
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
#--------------------------------------------------
mat = sio.loadmat('train.mat')
train_x1 = np.array(mat['x1'],dtype=np.float32)
train_x2 = np.array(mat['x2'],dtype=np.float32)
coef_train=np.ones((train_x1.shape[0],1))
train_y = np.array(mat['y'],dtype=np.int_)

train_X=np.concatenate((coef_train,train_x1,train_x2),axis=1)

model=keras.Sequential([keras.layers.Flatten(input_shape=(3,)),
	keras.layers.Dense(3,activation=tf.nn.relu,use_bias=True,bias_initializer='zeros'),
	keras.layers.Dense(2,activation=tf.nn.relu,use_bias=True,bias_initializer='zeros'),
	keras.layers.Dense(1,activation=tf.nn.sigmoid)])
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_X,train_y,epochs=50, batch_size=1)

#------------------------------------------------------------------
test = sio.loadmat('test.mat')
test_x1 = np.array(test['x1'],dtype=np.float32)
test_x2 = np.array(test['x2'],dtype=np.float32)
test_y = np.array(test['y'],dtype=np.int_)
coef_test=np.ones((test_x1.shape[0],1))
test_X=np.concatenate((coef_test,test_x1,test_x2),axis=1)

test_loss, test_acc= model.evaluate(test_X,test_y)
print('Test accuracy:',test_acc)
test_err=(1-test_acc)*100
print("Test error:{0:.2f}%".format(test_err))

print(model.get_weights())
