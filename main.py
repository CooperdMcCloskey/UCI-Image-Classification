import matplotlib.pyplot as plt
import tensorflow as tf

import config
from data import data_import
from training.model import get_model

dataset = data_import.createDataset(config.training_data_paths)
validation_dataset = data_import.createDataset(config.validation_data_paths)

model = get_model()
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
history = model.fit(dataset, epochs=config.epochs, validation_data=validation_dataset, class_weight=config.class_weights)


