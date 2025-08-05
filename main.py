import tensorflow as tf

import config
from data import data_import
from training.model import get_night_model, get_day_model
from training.PrintPredictionsCallback import PrintPredictionsCallback
from training.get_class_weights import get_class_weights

print("SETTINGS ---")
print(f" >> Classification type: {config.classification_type}")
print(f" >> Time: {config.time}")

dataset = data_import.createDataset(config.training_data_paths)
validation_dataset = data_import.createDataset(config.validation_data_paths, True)
class_weights = get_class_weights(dataset)

callbacks = []
if config.debugging:
  callbacks.append(PrintPredictionsCallback(validation_dataset))

model = get_day_model() if config.is_day else get_night_model()
model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
if(config.classification_type == 'general'):
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
