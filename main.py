import matplotlib.pyplot as plt
import tensorflow as tf
import math

import config
from data import data_import
from training.model import get_model
from training.PrintPredictionsCallback import PrintPredictionsCallback
from training.get_class_weights import get_class_weights


dataset, train_sample_count = data_import.createDataset(config.training_data_paths)
validation_dataset, validation_sample_count = data_import.createDataset(config.validation_data_paths)

class_weights = get_class_weights(dataset)

callbacks = []
if config.debugging:
  callbacks.append(PrintPredictionsCallback(validation_dataset))

model = get_model()
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
if(config.setting == 'general'):
  model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=['accuracy'])
else:
  model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
history = model.fit(
  dataset,
  epochs=config.epochs,
  validation_data=validation_dataset,
  class_weight=class_weights,
  callbacks=callbacks
)
