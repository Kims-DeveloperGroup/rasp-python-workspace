import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
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
	print(f'label={labels}\n')
	print(f'features={features}\n')
	
	#Preprocessing dataset
	normalizer = layers.Normalization()
	normalizer.adapt(features.copy())
	
	# Make regression model
	if model is None:
		model = tf.keras.Sequential([
	  	normalizer,
	  	layers.Dense(64, activation='relu'),
	  	layers.Dense(1)
		])
	
		model.compile(loss = tf.keras.losses.MeanSquaredError(),
	                      optimizer = tf.keras.optimizers.Adam())
	
	# Train model
	model.fit(features, labels, epochs=epochs)
	model.save(model_path)

path = '/Users/rica/Documents/data_v3.csv'
model_path = 'tf/models/test_v1.keras'
#train(path, model_path)
