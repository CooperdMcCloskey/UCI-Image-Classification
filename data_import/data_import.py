import csv
import os

# converts species code to numberic value for training (0 is false trigger and -1 is error)
label_map = {
    'CAFA': 1,
    'PRLO': 2,
    'HOSA': 3,
    'SCIU': 4,
    'BIRD': 5,
}

# turns the spreadsheet data into a filepath, label pair and puts it in the key.csv file.
def createKey(path):

  # extract spreadsheet data
  data = []
  with open(path+'/labels.csv', 'r') as file:
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
  image_count = sum(1 for image in os.scandir(path+'/photos') if image.is_file())

  # creates an numberic array of keys based on the label map
  key = [0] * image_count
  for i in range(len(labeled_photo_indices)):
    encoded_label = label_map.get(data[i+1][7], -1)
    if encoded_label == -1: print(f'Unknown label: {data[i+1][7]}, - IMG_{i+1}')
    key[labeled_photo_indices[i]] = encoded_label

  
  # writes the key array and corresponding image filepath to the csv file
  with open(path+'/key.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'label'])  # header

    for i in range(len(key)):
      if key[i] == -1: continue
      writer.writerow([f'{path}/IMG_{i+1:04d}', key[i]]) 


createKey('data_import/2022_11_09_BonitaCanyon1')
