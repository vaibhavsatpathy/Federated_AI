import numpy as np
import keras
from keras.datasets import mnist

no_batches=100

def data_prep():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	master_x=[]

	for i in range(len(np.unique(y_train))):
		temp=[]
		master_x.append(temp)

	for i in range(len(y_train)):
		master_x[y_train[i]].append(x_train[i])

	final_x=[]
	final_y=[]
	curr_ind=0
	for k in range(no_batches):
		X=[]
		Y=[]
		for i in range(len(np.unique(y_train))):
			data_to_read=int(len(master_x[i])/no_batches)
			temp_x=[]
			temp_y=[]
			for j in range(curr_ind,curr_ind+data_to_read):
				if j<(len(master_x[i])):
					val=master_x[i][j].reshape(28,28,1)
					temp_x.append(val)
					temp_y.append(np.asarray(i))
				else:
					pass
			temp_x=np.asarray(temp_x)
			temp_y=np.asarray(temp_y)
			X.extend(temp_x)
			Y.extend(temp_y)
		X=np.asarray(X)
		Y=np.asarray(Y)
		final_x.append(X)
		final_y.append(Y)
		curr_ind+=data_to_read

	return final_x,final_y