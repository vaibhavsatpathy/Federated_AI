import os
import sys
import Model_pb2
from keras.models import load_model
import numpy as np

###### you can also read the specs from coremltools.models.utils.load_model.get_spec

model_arc=load_model('/Users/vsatpathy/Desktop/Nudges POC/nudges/nudges_dev__2_new.h5')

model_1 = Model_pb2.Model()
model_2 = Model_pb2.Model()

master_weights_1=[]
master_bias_1=[]
master_weights_2=[]
master_bias_2=[]

with open("/Users/vsatpathy/Desktop/Nudges POC/nudges/nudges_dev__1_new.mlmodel", "rb") as f:

	model_1.ParseFromString(f.read())
	print(model_1)
	
	for layer in model_1.neuralNetworkClassifier.layers:
		layer_weights=layer.innerProduct.weights
		layer_bias=layer.innerProduct.bias
		master_weights_1.append(list(layer_weights.floatValue))
		master_bias_1.append(list(layer_bias.floatValue))

with open("/Users/vsatpathy/Desktop/Nudges POC/nudges/nudges_dev__2_new.mlmodel", "rb") as g:

	model_2.ParseFromString(g.read())
	print(model_2)
	
	for layer in model_2.neuralNetworkClassifier.layers:
		layer_weights=layer.innerProduct.weights
		layer_bias=layer.innerProduct.bias
		master_weights_2.append(list(layer_weights.floatValue))
		master_bias_2.append(list(layer_bias.floatValue))

final_weights=[]
for i in range(len(master_weights_1)):
	if len(master_weights_1[i])>0 and len(master_weights_2[i])>0:
		fin_weights=np.divide(np.add(master_weights_1[i],master_weights_2[i]),2)
		final_weights.append(fin_weights)

final_bias=[]
for i in range(len(master_bias_1)):
	if len(master_bias_1[i])>0 and len(master_bias_2[i])>0:
		fin_bias=np.divide(np.add(master_bias_1[i],master_bias_2[i]),2)
		final_bias.append(fin_bias)

shape_weights=[]
shape_bias=[]
for layer in model_arc.layers:
	g=layer.get_config()
	h=layer.get_weights()
	for i in range(len(h)):
		if i%2==0:
			shape_weights.append(h[i].shape)
		else:
			shape_bias.append(h[i].shape)

finale=[]
for i in range(len(final_weights)):
	temp_weights=np.reshape(final_weights[i],(shape_weights[i][0],shape_weights[i][1]))
	temp_bias=np.reshape(final_bias[i],(shape_bias[i]))
	finale.append(temp_weights)
	finale.append(temp_bias)
finale=np.asarray(finale)

model_arc.set_weights(finale)
model_arc.save('/Users/vsatpathy/Desktop/Nudges POC/nudges/fed_model.h5')