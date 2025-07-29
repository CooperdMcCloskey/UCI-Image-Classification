import csv
import os

import cv2
import numpy as np
import tensorflow as tf

import config

# turns the spreadsheet data into a filepath, label pair and puts it in the key.csv file.
def createKey(path, validation=False):

  # extract spreadsheet data
  data = []
  with open(path+'/data.csv', 'r') as file:
    csvreader = csv.reader(file)
    next(csvreader)
    for row in csvreader:
      data.append(row)
  
  # False triggers are not labeled in the spreadsheet so it is necessary to keep track of the photos for which we have a label.
  labeled_photo_indices = []
  for row in data[1:]:
    photo_number = int(''.join(filter(str.isdigit, row[4])))-1
    labeled_photo_indices.append(photo_number)

  # calculates the number of images in the folder
  image_count = sum(1 for image in os.scandir(path+'/photos') if image.is_file() and image.name.endswith('.JPG'))

  # creates an numberic array of keys based on the label map
  key = [0] * image_count
  for i in range(len(labeled_photo_indices)):
    encoded_label = config.label_map.get(data[i+1][7], -1)
    if encoded_label == -1: print(f'Unknown label: {path} - {data[i+1][7]}, - IMG_{i}')
    key[labeled_photo_indices[i]] = encoded_label

    #removes unsure photos from pool
    if int(data[i+1][5]) > 2:
      key[labeled_photo_indices[i]] = -1

  # writes to the csv file
  with open(path+'/key.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label', 'is_day'])  # header

    # loops through every image
    for i in range(image_count):
      if key[i] == -1 or key[i] == -2: continue

      image_path = f'{path}/photos/IMG_{i+1:04d}.JPG'

      img = cv2.imread(image_path) # this section is very inefficient and can be greatly optimized by running in parallel or through other means.
      is_day = is_color(img)

      writer.writerow([image_path, key[i], is_day]) 
  

def load_image(path, label, validation = False):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, config.image_size)
  image = tf.image.rgb_to_grayscale(image) # most photos are at night and black and white anyways so this is likely to improve performance
  image = image / 255.0 

  # image augmentation to avoid overfitting - can be removed if overfitting is not a problem
  if not validation:
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.adjust_contrast(image, 2.0)
  return image, label

def is_color(image):
  b, g, r = cv2.split(image)
  diff_rg = np.abs(r.astype(int) - g.astype(int))
  diff_gb = np.abs(g.astype(int) - b.astype(int))
  diff_rb = np.abs(r.astype(int) - b.astype(int))

  max_diff = np.maximum(np.maximum(diff_rg, diff_gb), diff_rb)
  mean_diff = np.mean(max_diff)

  return mean_diff > 5


def createDatasets(paths, validation = False):
  # extracts all image paths and all keys from all the given directories
  image_paths = []
  image_labels = []

  for path in paths:
    with open(path+'/key.csv', 'r') as file:
      csvreader = csv.reader(file)
      next(csvreader)
      for row in csvreader:
        image_path = row[0]
        image_label = int(row[1])
        is_day = row[2].strip().lower() in ('true', '1')
        if is_day == config.is_day:
          image_paths.append(image_path)
          image_labels.append(image_label)

  # creates tensorflow datasets from the data
  path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
  label_ds = tf.data.Dataset.from_tensor_slices(image_labels)
  ds = tf.data.Dataset.zip((path_ds, label_ds))

  ds = ds.map(lambda path, label: load_image(path, label, validation), num_parallel_calls=tf.data.AUTOTUNE) # calls load_image for all the images in the dataset in parralel
  ds.cache()
  ds = ds.shuffle(buffer_size=config.data_shuffle_buffer_size, reshuffle_each_iteration=True) # randomizes dataset order
  ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

  return ds
  
