import pandas as pd
import numpy as np

import os
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')

labelNames = ['Jab', 'Straight', 'FR-Hook', 'BH-Hook', 'FR-Upper', 'BH-Upper', 'FR-Body', 'BH-Body    ', 'BodyJab', 'BodyStraight']

active_landmark_indices = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
def train(
		dataset_path, 
		model_path,
		epochs,
		test_dataset_path,
		learning_rate,
		dropout_rate= 0.2,
		batch_size = 100,
		regular_rate = 0.0,
	):
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
	trained_model = _train(
						features,
						labels,
						test_features= test_feats,
						test_labels=test_labels,
						epochs=epochs,
						batch_size = batch_size,
						model=model,
						learning_rate = learning_rate,
						dropout_rate = dropout_rate,
						regular_rate = regular_rate,
					)
	trained_model.save(model_path)
	trained_model.summary()
	return model

class TrainCallback(tf.keras.callbacks.Callback):
	def __init__(self):
		self.p1 = plt.subplots()[1]
		self.p2 = plt.subplots()[1]
		self.x = []
		self.acc = []
		self.loss = []
		self.val_acc = []
		self.val_loss = []

	def on_epoch_end(self, epoch, logs={}):
		val_loss = logs['val_loss'] 
		val_acc = logs['val_categorical_accuracy']
		train_data_acc = logs['categorical_accuracy']
		train_data_loss = logs['loss']
		self.x.append(epoch)
		self.acc.append(train_data_acc)
		self.loss.append(train_data_loss)
		self.val_loss.append(val_loss)
		self.val_acc.append(val_acc)
		if(val_acc >= 0.90) and (train_data_acc > 0.80):
			self.model.stop_training = True
	
	def on_train_end(self, logs=None):
		print('훈련 끝')
		self.p1.plot(self.x, self.val_loss,'g', linewidth=2)
		self.p1.plot(self.x, self.loss,'k', linewidth=2)

		self.p2.plot(self.x, self.acc, 'k', linewidth=2)
		self.p2.plot(self.x, self.val_acc,'g', linewidth=2)
		plt.show()

def _train(
		features,
		labels, 
		test_features,
		test_labels,
		learning_rate,
		batch_size,
		dropout_rate,
		regular_rate,
		epochs= 10,
		model = None,
	):
	features_copy = features.copy()
	features_copy = features_copy.reshape((-1, 33,3))
	test_features= test_features.reshape((-1, 33,3))
	onehot_enc = tf.one_hot(indices=labels, depth=10, dtype = tf.float32)
	test_onehot_enc = tf.one_hot(indices=test_labels, depth=10, dtype = tf.float32)
	
	features_copy = features_copy.take(active_landmark_indices, axis=1)
	test_features = test_features.take(active_landmark_indices, axis=1)
	normalizer = layers.Normalization(axis=-1)
	normalizer.adapt(features_copy)

	# Make regression model
	if model is None:
		model = tf.keras.Sequential([
	  	normalizer,
	  	layers.Flatten(),
	  	layers.Dense(10, activation='relu', kernel_regularizer = regularizers.L2(regular_rate)),
		layers.Dropout(rate = dropout_rate),
	  	layers.Dense(10, activation='softmax'),
		])
	
		model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
	                      optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate),
						  metrics = [tf.keras.metrics.CategoricalAccuracy()])
	# Train model
	model.fit(
				x=features_copy,
				y=onehot_enc,
				epochs=epochs,
				#validation_split = 0.2,
				validation_data = (test_features, test_onehot_enc),
				batch_size = batch_size,
				callbacks= [
					TrainCallback(),
					callbacks.EarlyStopping(
						#monitor='val_categorical_accuracy',
						monitor='val_loss',
						patience=100,
						restore_best_weights = True,
						start_from_epoch = 350,
						verbose = 1,
					),
				],
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

