import os
import argparse

import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
# from sklearn.preprocessing import LabelEncoder
# from keras.layers import BatchNormalization, Conv2D, MaxPool2D, Flatten, Dense, Dropout
# from keras.utils.io_utils import HDF5Matrix
# from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras import applications
from keras.layers import Input
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras.utils.training_utils import multi_gpu_model

def top_5_accuracy(x,y): 
	return top_k_categorical_accuracy(x,y, 5)

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='quickDraw classifier')
    parser.add_argument('-g', '--G', type=int, default=1)
    parser.add_argument('-p', '--path')
    parser.add_argument('-n', '--name')
    args = parser.parse_args()
    globals().update(vars(args))

    print("[INFO] GPU devices:%s" % get_available_gpus())

	if G <= 1:
		print("[INFO] training with 1 GPU...")
		model = applications.mobilenetv2.MobileNetV2(
			include_top=True, classes=345, weights=None, input_tensor=Input(shape=(64,64,1))
			)

	# otherwise, we are compiling using multiple GPUs
	else:
		print("[INFO] training with {} GPUs...".format(G))
	 
		# we'll store a copy of the model on *every* GPU and then combine
		# the results from the gradient updates on the CPU
		with tf.device("/cpu:0"):
			# initialize the model
			model = applications.mobilenetv2.MobileNetV2(
				include_top=True, classes=345, weights=None, input_tensor=Input(shape=(64,64,1))
				)
		
		# make the model parallel
		model = multi_gpu_model(model, gpus=G)

	adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.0, clipnorm=5)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=["accuracy", top_5_accuracy])

	earlystop = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(save_path+'/%s/checkpoint_model.h5'%model_name, monitor='val_loss', verbose=1, 
	                 save_best_only=True, mode='min', save_weights_only = False)
	reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)

	print("[INFO] training network...")
	H = model.fit_generator(reader.run_generator(train_names),
	        steps_per_epoch=train_steps, epochs=config.nbepochs, shuffle=False, verbose=1,
	        validation_data=reader.run_generator(test_names), validation_steps=test_steps, 
	        use_multiprocessing=False, workers=1, callbacks=[checkpoint, earlystop, reduceLROnPlat])
	# H = model.fit_generator(
	# 	aug.flow(trainX, trainY, batch_size=64 * G),
	# 	validation_data=(testX, testY),
	# 	steps_per_epoch=len(trainX) // (64 * G),
	# 	epochs=NUM_EPOCHS,
	# 	callbacks=callbacks, verbose=2)

	pickle.dump(H.history, open(save_path+'/'+model_name+'/loss_history_fold#%i.pickle.dat'%fold, 'wb'))

	model_json = model.to_json()
	with open(save_path+'/'+model_name+"/model.json", "w") as json_file:
	    json_file.write(model_json)
	print("[INFO] Finished!")