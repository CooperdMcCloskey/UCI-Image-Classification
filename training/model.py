import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models # type: ignore
import config

def get_model():

  model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=config.image_size + (1,)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(len(config.label_map)+1, activation='softmax')  
])

  return model