import pandas as pd
import numpy as np

import os
import time
import tensorflow as tf
from tensorflow.keras import layers

labelNames = ['Jab', 'Straight', 'FR-Hook', 'BH-Hook', 'FR-Upper', 'BH-Upper', 'FR-Body', 'BH-Body    ', 'BodyJab', 'BodyStraight']

active_landmark_indices = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def train(dataset_path, model_path, epochs, test_dataset_path):
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
	test_feats, test_labels = _loadDataset(test_dataset_path)
	trained_model = _train(features, labels, test_features= test_feats, test_labels=test_labels, epochs=epochs, model=model)
	trained_model.save(model_path)
	trained_model.summary()
	return model

class TrainCallback(tf.keras.callbacks.Callback):
	def __init__(self, m, f, l):
		self.m = m
		self.f = f
		self.l = l

	def on_epoch_end(self, epoch, logs={}):
		loss, accuracy = self.m.evaluate(self.f, tf.one_hot(indices=self.l, depth=10, dtype = tf.float32))
		print(f'EPOCH {epoch + 1}: test_accuracy={accuracy} test_loss={loss}')
		if(accuracy >0.80) or (loss > 10.0 and epoch > 100):
			print(f'RESULT 정확도:{accuracy} 손실: {loss}')
			self.model.stop_training = True

def _train(features, labels, test_features, test_labels, epochs= 10, model = None):
	features_copy = features.copy()
	features_copy = features_copy.reshape((-1, 33,3))
	test_features= test_features.reshape((-1, 33,3))
	onehot_enc = tf.one_hot(indices=labels, depth=10, dtype = tf.float32)
	
	features_copy = features_copy.take(active_landmark_indices, axis=1)
	test_features = test_features.take(active_landmark_indices, axis=1)
	normalizer = layers.Normalization(axis=-1)
	normalizer.adapt(features_copy)

	# Make regression model
	if model is None:
		model = tf.keras.Sequential([
	  	normalizer,
	  	layers.Flatten(),
	  	layers.Dense(10, activation='relu'),
		layers.Dropout(rate=0.2),
	  	layers.Dense(10, activation='softmax'),
		])
	
		model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
	                      optimizer = tf.keras.optimizers.Adam(),
						  metrics = [tf.keras.metrics.CategoricalAccuracy()])
	# Train model
	model.fit(
				x=features_copy,
				y=onehot_enc,
				epochs=epochs,
				validation_split = 0.2,
				batch_size = 20,
				callbacks= TrainCallback(model, test_features, test_labels)
			)
	return model

def _test(features, labels, model):
	features = np.reshape(features, (-1, 33, 3))
	expected = labels
	features= features.take(active_landmark_indices, axis=1)
	actual = model.predict(features)
	loss, accuracy = model.evaluate(features, tf.one_hot(indices=labels, depth=10, dtype = tf.float32))
	copied_features = features.copy()
	error_feats = []
	error_labels = []
	for idx, value in enumerate(actual):
		flag = ''
		if expected[idx] != np.argmax(value):
			flag = '[INCORRECT]'
			error_feats.append(features[idx])
			error_labels.append(expected[idx])
			print(f'{flag} {idx}. expected={labelNames[expected[idx]]} predicted={labelNames[np.argmax(value)]},\n values={value}')
	print(f'total={len(features)} errors={len(error_feats)}')
	return np.array(error_feats), np.array(error_labels)
def _loadDataset(path):
	dataset = pd.read_csv(path).sort_values(by='label', axis= 0)
	labels = np.array(dataset.pop("label"))
	features = np.array(dataset)
	return features, labels

def loadFileAndTest(dataset_path, model_path):
	model = tf.keras.models.load_model(model_path)
	features, expected = _loadDataset(dataset_path)	
	error = _test(features, expected, model)
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

