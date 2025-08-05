import csv
import os

import tensorflow as tf

import config 

#creates an image of all the average false triggers to serve as a baseline.
def create_baseline_image(path, is_day):
  images = []
  with open(path+'/key.csv', 'r') as file:
      csvreader = csv.reader(file)
      next(csvreader)
      for row in csvreader:
         image_is_day = row[2].strip().lower() in ('true', '1')
         if int(row[1]) == 0 and image_is_day == is_day:
          images.append(load_image(row[0]))

  image_stack = tf.stack(images, axis=0)
  baseline_image = tf.reduce_mean(image_stack, axis=0)

  time = 'day' if is_day else 'night'
  image_uint8 = tf.image.convert_image_dtype(baseline_image, dtype=tf.uint8)
  encoded_image = tf.io.encode_jpeg(image_uint8)
  tf.io.write_file(f'{path}/{time}BaselinePhoto.jpeg', encoded_image)

def load_image(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, config.image_size)
  image = image / 255.0 
  image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-6) # boosts contrast in the image, stretching values so that the minimum value is 0 and the maximum is 1, 1e-6 to prevent divide by zero errors.
  return image


class ImageLoader:
  def __init__(self):
    self.time = 'day' if config.is_day else 'night'

    keys = []
    images = []

    for folder_path in config.training_data_paths:
      key = f'{folder_path}_{self.time}'
      image = load_image(f'{folder_path}/{self.time}BaselinePhoto.jpeg')
      keys.append(key)
      images.append(image)

    # Turn list of keys into tf.constant
    self.keys_tensor = tf.constant(keys)
    self.values_tensor = tf.stack(images)

    # Create hash table: maps folder_key -> index in values_tensor
    initializer = tf.lookup.KeyValueTensorInitializer(
      keys=self.keys_tensor,
      values=tf.range(len(images), dtype=tf.int32)
    )
    self.baseline_images = tf.lookup.StaticHashTable(initializer, default_value=-1)


  def process_image(self, path, label, validation = False):
    image = load_image(path)

    # Extract folder path
    path_parts = tf.strings.split(path, '/')
    folder_name = tf.strings.reduce_join(path_parts[:-2], separator='/')
    folder_key = tf.strings.join([folder_name, self.time], separator='_')

    # Lookup index in values_tensor
    index = self.baseline_images.lookup(folder_key)
    baseline_image = tf.gather(self.values_tensor, index)

    image = image - baseline_image # removes the background from the image, giving a new image in the range [-1, 1]
    image = tf.abs(image) # moves the image from [-1,1] to [0, 1], setting background to black and everything else to light
    image = tf.where(image < config.noise_reduction_threshold, 0.0, image) # removes noise
    image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-6) # boost contrast
    image = tf.pow(image, 1.5) # boost contrast
    
    # image augmentation to avoid overfitting - can be removed if overfitting is not a problem
    # if not validation:
    #   image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label