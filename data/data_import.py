import csv
import os

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

    #removes unsure photos from validation pool
    if validation and int(data[i+1][5]) == 3:
      key[labeled_photo_indices[i]] = -1

  # writes the key array and corresponding image filepath to the csv file
  with open(path+'/key.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])  # header

    for i in range(len(key)):
      if key[i] == -1 or key[i] == -2: continue
      writer.writerow([f'{path}/photos/IMG_{i+1:04d}.JPG', key[i]]) 


def load_image(path, label):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, config.image_size)
  image = tf.image.rgb_to_grayscale(image) # most photos are at night and black and white anyways so this is likely to improve performance
  image = image / 255.0 
  return image, label


def createDataset(paths):
  # extracts all image paths and all keys from all the given directories
  image_paths = []
  image_labels = []
  for path in paths:
    with open(path+'/key.csv', 'r') as file:
      csvreader = csv.reader(file)
      next(csvreader)
      for row in csvreader:
        image_paths.append(row[0])
        image_labels.append(int(row[1]))

  

  # creates tensorflow datasets from the data
  path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
  label_ds = tf.data.Dataset.from_tensor_slices(image_labels)
  ds = tf.data.Dataset.zip((path_ds, label_ds))

  
  ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE) # calls load_image for all the images in the dataset in parralel
  ds = ds.shuffle(buffer_size=config.data_shuffle_buffer_size) # randomizes dataset order
  ds = ds.batch(config.batch_size).prefetch(tf.data.AUTOTUNE)

  return ds
  
