import pandas as pd
import numpy as np

import os
import time
import tensorflow as tf
from tensorflow.keras import layers

labelNames = ['Jab', 'Straight', 'FR-Hook', 'BH-Hook', 'FR-Upper', 'BH-Upper', 'FR-Body', 'BH-Body    ', 'BodyJab', 'BodyStraight']

def _normalize(poses, truncate_feat=True):
     poses=poses.reshape((-1, 33,3))
     normalized_poses =  []
     for pose in poses:
         landmarks = None     
         if truncate_feat == True:
             landmarks = pose[11:]
         else:
             landmarks = pose
         #11 left shoulder 
         minX = landmarks[11][0]
         minY = landmarks[11][1]
         minZ = landmarks[11][2]
         normalized_landmarks = []
         for landmark in landmarks:
             normalized = landmark.copy()
             #print(f'before={normalized}')
             normalized[0] = landmark[0] - minX
             normalized[1] = landmark[1] - minY
             normalized[2] = landmark[2] - minZ
             #print(f'after={normalized}')
             normalized_landmarks.append(normalized.tolist())
         normalized_poses.append(normalized_landmarks)
     return np.array(normalized_poses)

def normalize_file(source, dest, truncate_feat=False):
    dataset = pd.read_csv(source).sort_values(by='label', axis= 0)
    
    labels= np.array(dataset.pop('label'))
    cols= dataset.columns.tolist()
    feats = np.array(dataset).reshape((-1, 33,3))
    
    normalized_feats = []
    for feat in _normalize(feats, truncate_feat):
        normalized_feats.append(feat.flatten())
    normalized_feats = np.array(normalized_feats)
    if truncate_feat == True:
        cols = cols[33:]
    dp = pd.DataFrame(data=normalized_feats, columns=cols)
    dp.insert(loc=0, column='label', value=labels)
    dp.to_csv(path_or_buf=dest, mode='w', index=False)

def train(dataset_path, model_path, epochs, normalize=False):
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
	trained_model = _train(features, labels, epochs, model=model, normalize=normalize)
	trained_model.save(model_path)
	trained_model.summary()
	return model

def _train(features, labels, epochs= 10, model = None, normalize=False):
	features_copy = features.copy()
	if normalize == True:
		features_copy= _normalize(poses= features_copy)
	else:
		features_copy= features_copy.reshape((-1, 33,3))
	onehot_enc = tf.one_hot(indices=labels, depth=10, dtype = tf.float32)
	
	normalizer = layers.Normalization(axis=-1)
	normalizer.adapt(features_copy)

	# Make regression model
	if model is None:
		model = tf.keras.Sequential([
		#layers.InputLayer(shape=(33,3)),
	  	normalizer,
	  	layers.Flatten(),
		layers.Dense(189, activation='relu'),
		layers.Dense(90, activation='relu'),
		layers.Dropout(rate=0.2),
	  	layers.Dense(45, activation='relu'),
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
				batch_size = 40,
			)
	return model

def _test(features, labels, model, normalize):
	if normalize == True:
		features = _normalize(features)
	else:
		features = np.reshape(features, (-1, 33, 3))
	expected = labels
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

def loadFileAndTest(dataset_path, model_path, normalize= False):
	model = tf.keras.models.load_model(model_path)
	dataset = pd.read_csv(dataset_path).sort_values(by='label', axis= 0)
	expected = np.array(dataset.pop("label"))
	features = np.array(dataset)
	error = _test(features, expected, model, normalize)
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

