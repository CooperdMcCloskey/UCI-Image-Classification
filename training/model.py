import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, initializers # type: ignore
import config

def get_day_model():
    pass

def get_night_model():
    inputs = keras.Input(shape=config.image_size + (1,))  # grayscale input

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.SpatialDropout2D(0.2)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Global feature pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = layers.Dense(64, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    # Output layer
    output_activation = 'sigmoid' if config.classification_type == 'general' else 'softmax'
    outputs = layers.Dense(config.class_count, activation=output_activation, bias_initializer=initializers.Constant(0) )(x)
    

    # Build the model
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model