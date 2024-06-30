import pandas as pd
import numpy as np

import os
import time
import tensorflow as tf
from tensorflow.keras import layers
def train(dataset_path, model_path, epochs):
	# Load csv data and make features and labels
	model = None
	try:
		model = tf.keras.models.load_model(model_path)
		print(f'The model is loaded in {model_path}')
	except:
		print(f'Newly creates in {model_path}')
	dataset = pd.read_csv(dataset_path)
	labels = np.array(dataset.pop("label"))
	features = np.array(dataset)
	trained_model = _train(features, labels, epochs, model=model)
	trained_model.save(model_path)
	trained_model.summary()
	return model
def _train(features, labels, epochs= 10, model = None):
	#Preprocessing dataset
	normalizer = layers.Normalization()
	normalizer.adapt(features.copy())
	onehot_enc = tf.one_hot(indices=labels, depth=10, dtype = tf.int32)
	# Make regression model
	if model is None:
		model = tf.keras.Sequential([
		layers.InputLayer(input_shape=(33 * 3,)),
	  	normalizer,
	  	layers.Dense(64, activation='relu'),
	  	layers.Dense(32, activation='relu'),
	  	layers.Dense(16, activation='relu'),
	  	#layers.Dense(10),
		layers.Dense(10, activation='softmax'),
		])
	
		model.compile(loss = tf.keras.losses.MeanAbsoluteError(),
	                      optimizer = tf.keras.optimizers.Adam())
	# Train model
	model.fit(features, onehot_enc, epochs=epochs)
	return model

def test(features, labels, model):
	expected = labels
	actual = model.predict(features)
	copied_features = features.copy()
	error_feats = []
	error_labels = []
	for idx, value in enumerate(actual):
		flag = ''
		if expected[idx] != np.argmax(value):
			flag = '[INCORRECT]'
			error_feats.append(features[idx])
			error_labels.append(expected[idx])
			print(f'{flag} expected={expected[idx]} predicted={np.argmax(value)},\n values={value}')
	return np.array(error_feats), np.array(error_labels)

def loadFileAndTest(dataset_path, model_path):
	model = tf.keras.models.load_model(model_path)
	dataset = pd.read_csv(dataset_path)
	expected = np.array(dataset.pop("label"))
	features = np.array(dataset)
	error = test(features, expected, model)
	model.summary()
	#return error

def trainAndTest(dataset_path, model_path, trainEpochs):
	model = train(dataset_path, model_path, trainEpochs)
	error_feats, error_labels = loadFileAndTest(dataset_path, model_path)
	# Re-train with error features
#	while len(error_feats) > 0:
#		model = _train(error_feats, error_labels, trainEpochs, model=model)
#		error_feats, error_labels = test(error_feats, error_labels, model)
#		time.sleep(10)

