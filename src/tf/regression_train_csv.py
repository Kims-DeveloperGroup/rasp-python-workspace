import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

# Load csv data and make features and labels
path = '/Users/rica/Documents/data.csv'
dataset = pd.read_csv(path)
dataset.head()
features = dataset.copy()
labels = features.pop('label')

print(features)
print(labels)

# Preprocessing dataset
features = np.array(features)
print(f'features={features}')
normalizer = layers.Normalization()
normalizer.adapt(features)

# Make regression model
model = tf.keras.Sequential([
  normalizer,
  layers.Dense(64, activation='relu'),
  layers.Dense(1)
])

model.compile(loss = tf.keras.losses.MeanSquaredError(),
                      optimizer = tf.keras.optimizers.Adam())
# Train model
model.fit(features, labels, epochs=10)
model.save('tf/models/test_v1.keras')
