from __future__ import print_function
import keras
from keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt 

# input image dimensions 28x28=784
img_rows, img_cols = 28, 28
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')  
x_test = x_test.astype('float32')
#to convert the value of element in each array to be less than 1
x_train /= 255    
x_test /= 255
#y_train:1D array with 60000 data
#y_test:1D array with 10000 data
#x_train:(60000, 28,28)
#x_test:(10000, 28, 28)
#print(x_train[0])
#print(x_train.shape[0])
#----------------------------------------------------
X_train=x_train.reshape(-1, img_rows*img_cols)    #convert 28x28 of 1 image to 1D of 784 elements        ##x_test[:1000]

train_filter0 = np.where((y_train==0))
train_filter1 = np.where((y_train==1))
train_filter2 = np.where((y_train==2))
train_filter3 = np.where((y_train==3))
train_filter4 = np.where((y_train==4))
train_filter5 = np.where((y_train==5))
train_filter6 = np.where((y_train==6))
train_filter7 = np.where((y_train==7))
train_filter8 = np.where((y_train==8))
train_filter9 = np.where((y_train==9))

X_train0,Y_train0=X_train[train_filter0], y_train[train_filter0]
X_train1,Y_train1=X_train[train_filter1], y_train[train_filter1]
X_train2,Y_train2=X_train[train_filter2], y_train[train_filter2]
X_train3,Y_train3=X_train[train_filter3], y_train[train_filter3]
X_train4,Y_train4=X_train[train_filter4], y_train[train_filter4]
X_train5,Y_train5=X_train[train_filter5], y_train[train_filter5]
X_train6,Y_train6=X_train[train_filter6], y_train[train_filter6]
X_train7,Y_train7=X_train[train_filter7], y_train[train_filter7]
X_train8,Y_train8=X_train[train_filter8], y_train[train_filter8]
X_train9,Y_train9=X_train[train_filter9], y_train[train_filter9]

first100x_0=X_train0[:100,:]
first100x_1=X_train1[:100,:]
first100x_2=X_train2[:100,:]
first100x_3=X_train3[:100,:]
first100x_4=X_train4[:100,:]
first100x_5=X_train5[:100,:]
first100x_6=X_train6[:100,:]
first100x_7=X_train7[:100,:]
first100x_8=X_train8[:100,:]
first100x_9=X_train9[:100,:]

first100y_0=Y_train0[:100]
first100y_1=Y_train1[:100]
first100y_2=Y_train2[:100]
first100y_3=Y_train3[:100]
first100y_4=Y_train4[:100]
first100y_5=Y_train5[:100]
first100y_6=Y_train6[:100]
first100y_7=Y_train7[:100]
first100y_8=Y_train8[:100]
first100y_9=Y_train9[:100]
#-----------------------------------------
amount= 50
lines = 5
columns = 10
#for checking number with the plots
#number = np.zeros(amount)
#for i in range(amount):
#    number[i] = first100y_0[i]

#fig = plt.figure()
#for i in range(amount):
#    ax = fig.add_subplot(lines, columns, 1 + i) #create 50 subplots and label them
#    plt.imshow(first100x_0.reshape(100,img_rows, img_cols)[i,:,:], cmap='binary')    #for plotting image #binary creates black and white picture
#    plt.sca(ax)           #for instance
#    ax.set_xticks([], [])
#    ax.set_yticks([], [])

#plt.show()
#--------------------------------------------------
#Normalization
Xtrain_mean=np.mean(X_train, axis=0)     #axis=0, take the mean of each column
Xtrain_std=np.std(X_train, axis=0)+1e-10
Xtrain_nor=(X_train-Xtrain_mean)/Xtrain_std #subtract vector from matrix #divide matrix by vector

Xtrain_nor_mean=np.mean(Xtrain_nor, axis=0)
Xtrain_nor_std=np.std(Xtrain_nor, axis=0)
#print(Xtrain_nor_mean[50]) #mean of normalized data is zero
#print(Xtrain_nor_std[50])  #std of normalized data is 1
cov_mat=np.matmul(Xtrain_nor.T, Xtrain_nor) #covariance matrix
#print(cov_mat.shape)
#-----------------------------------------------------------
#Compute eigen vectors
eig_vals,eig_vecs=np.linalg.eig(cov_mat)
#print(eig_vecs.shape)
eig_pairs=[(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
#print(eig_pairs[0][1].shape)                      #eig_pairs[0] means the first pair,eig_pair[0][0] means its eig_value,eig_pair[0][1] means its eigen vector
eig_pairs.sort(key=lambda x:x[0], reverse=True)   #reverse = true means from big to small #x[0] sort based on first indicator, which is eigen value
#--------------------------------------------------------------------------------
keep_dim =50
D=np.array([eig_pairs[i][1] for i in range(keep_dim)], dtype=np.float32) #eig_pairs[][1] 1 means selecting eig_vector only #we only select the eig_vector which corresponds to the largest eig_value #dimension remains the same here
#print(D.shape)
#print(Xtrain_nor.shape)
Xtrain_nor_enc= np.matmul(Xtrain_nor, D.T)   #size becomes 60000x500
Xtrain_nor_dec= np.matmul(Xtrain_nor_enc, D) #size becomes 60000x784

Xtrain_dec = (Xtrain_nor_dec*Xtrain_std)+Xtrain_mean       #transform normalized data into the original range (z=x-mean/sigma)
Xtrain_dec_rec =Xtrain_dec.reshape(-1, img_rows, img_cols) 
#print the result
fig=plt.figure()
for i in range(amount):
	ax = fig.add_subplot(lines, columns, 1+i)
	plt.imshow(Xtrain_dec_rec[i,:,:], cmap='binary')
	plt.sca(ax)
	ax.set_xticks([],[])
	ax.set_yticks([],[])
#plt.xlabel()
plt.show()	

