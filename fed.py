from dev_1 import train_1
from dev_2 import train_2
from models import architecture

import os
import numpy as np

import keras
from keras.models import load_model

def fed_avg(master_weights):

	temp_weights=master_weights[0]
	for j in range(1,len(master_weights)):
		temp_weights=np.add(temp_weights,master_weights[j])
	final_weights=np.divide(temp_weights,len(master_weights))

	return final_weights

no_of_master_epochs=1
for i in range(no_of_master_epochs):
	train_1(i)
	train_2(i+no_of_master_epochs)

	model_arc=architecture()
	path='/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/dev_models/'
	all_local_models=sorted(os.listdir(path))

	master_weights=[]
	for i in range(1,len(all_local_models)):
		model=load_model(path+all_local_models[i])
		local_weights=[]
		for layer in model.layers:
			h=layer.get_weights()
			for j in range(len(h)):
				local_weights.append(h[j])
		master_weights.append(local_weights)

	final_weights=fed_avg(master_weights)

	model_arc.set_weights(final_weights)
	model_arc.save('/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/master_model/master_model.h5')

	print('MASTER MODEL SAVED')