import os
import sys
import Model_pb2
from keras.models import load_model
import numpy as np

model=Model_pb2.Model()
path='/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/dev_mlmodels/'
no_master_epochs=2
comp_layers=['activation','flatten','max']

def fed_averaging(master_weights,master_bias,shape_weights,shape_bias):
	temp_weights=master_weights[0]
	temp_bias=master_bias[0]
	for j in range(1,len(master_weights)):
		temp_weights=np.add(temp_weights,master_weights[j])
		temp_bias=np.add(temp_bias,master_bias[j])
	final_weights=np.divide(temp_weights,len(master_weights))
	final_bias=np.divide(temp_bias,len(master_bias))

	updated_weights=[]
	for i in range(len(shape_weights)):
		up_weights=np.reshape(final_weights[i],shape_weights[i])
		up_bias=np.reshape(final_bias[i],shape_bias[i])
		updated_weights.append(up_weights)
		updated_weights.append(up_bias)
	updated_weights=np.asarray(updated_weights)

	return updated_weights

def main():

	all_mlmodel=sorted(os.listdir(path))

	master_weights=[]
	master_bias=[]
	for i in range(1,len(all_mlmodel)):
		model_weights=[]
		model_bias=[]
		with open(path+all_mlmodel[i],'rb') as f:
			model.ParseFromString(f.read())
			for layer in model.neuralNetworkClassifier.layers:
				sub_names=layer.name.split('_')
				counter=0
				for word in comp_layers:
					if word in sub_names:
						counter+=1
				if counter==0 and sub_names[0]=='dense':
					layer_weights=layer.innerProduct.weights
					model_weights.append(np.asarray(list(layer_weights.floatValue)))

					layer_bias=layer.innerProduct.bias
					model_bias.append(np.asarray(list(layer_bias.floatValue)))
					#print(sub_names)
				elif counter==0 and sub_names[0]=='conv2d':
					layer_weights=layer.convolution.weights
					model_weights.append(np.asarray(list(layer_weights.floatValue)))

					layer_bias=layer.convolution.bias
					model_bias.append(np.asarray(list(layer_bias.floatValue)))
					#print(sub_names)

		master_weights.append(model_weights)
		master_bias.append(model_bias)
	#print(len(master_weights[1][0]))

	model_arc=load_model('/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/master_model/master_model.h5')

	shape_weights=[]
	shape_bias=[]
	for layer in model_arc.layers:
		g=layer.get_config()
		h=layer.get_weights()
		split_name=g['name'].split('_')
		#print(split_name)
		counter=0
		for word in comp_layers:
			if word in split_name:
				counter+=1
		#print(counter)
		if counter==0:
			for i in range(len(h)):
				if i%2==0:
					shape_weights.append(h[i].shape)
				else:
					shape_bias.append(h[i].shape)

	updated_weights=fed_averaging(master_weights,master_bias,shape_weights,shape_bias)
	model_arc.set_weights(updated_weights)

main()