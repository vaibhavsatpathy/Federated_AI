import numpy as np
import os
from models import architecture
from data import data_prep
import coremltools

import keras
from keras import backend as K
from keras.models import load_model

def testing(model,x):
	test_inp=x[0][100]
	test_inp=np.expand_dims(test_inp,axis=0)
	result=model.predict(test_inp)
	print(np.argmax(result[0]))

def train_1(nth_epoch):

	img_rows=28
	img_cols=28

	batch_size=8
	epochs=5

	x,y=data_prep()
	labels=['0','1','2','3','4','5','6','7','8','9']

	if len(os.listdir('/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/master_model'))==1:
		model=architecture()
		#model.summary()
	else:
		model=load_model('/Users/vsatpathy/Desktop/Nudges POC/fed_ai/master/master_model/master_model.h5')
		#model.summary()

	model.fit(x[nth_epoch],y[nth_epoch],batch_size=batch_size,epochs=epochs,verbose=1)
	model.save('dev_models/model_1.h5')

	coreml_model = coremltools.converters.keras.convert('dev_models/model_1.h5',class_labels = labels)
	coreml_model.save('dev_mlmodels/model_1.mlmodel')