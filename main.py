import matplotlib.pyplot as plt
import tensorflow as tf

import config
from data import data_import
from training.model import get_model

data_import.createKey('data/2022_11_09_BonitaCanyon1')

dataset = data_import.createDataset(['data/2022_11_09_BonitaCanyon1'])

model = get_model()
model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
history = model.fit(dataset, epochs=config.epochs)


