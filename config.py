# converts species code to numberic value for training (0 is false trigger and -1 is error and do not need to be included in the map)
label_map = {
  'UNK': -2,
  '#N/A': -2,
  'VEHI': 0, # vehicle - counted as false trigger
  'BIKE': 0, # bike - counted as false trigger
  'HOSA': 0, # human - counted as false trigger
  'CAFA': 1, # coyote
  'LYRU': 2, # bobcat
  'RASP': 3, # rabbit
  'PRLO': 4, # raccoon
  'BIRD': 5, # bird
  'DIVI': 6, # opossum
  'MEME': 7, # skunk
  'SCIU': 8, # squirrel
  'OTOS': 8, # ground squirrel - grouped with squirrels
  'NEOSP': 9, # wood rat
}

class_count = 10 # 0-9 is 10 different classes

# Because certain classes are less common, it is necessary to punish the model more severely when it fails to recognize on of the rare classes
# run countLabels.py to inform the weighing of classes
class_weights = {
  0: 1, # false trigger - keep at 1
  1: 5, # coyote
  2: 1, # bobcat - not necessary to punish as there is such a limited number of bobcat photos it is impossible to learn
  3: 20, # rabbit
  4: 10, # raccoon
  5: 10, # bird
  6: 20, # opossum
  7: 1, # skunk - not necessary to punish as there is such a limited number of bobcat photos it is impossible to learn
  8: 10, # squirrel
  9: 1, # wood rat - not necessary to punish as there is such a limited number of bobcat photos it is impossible to learn
}

training_data_paths = [
  'data/2023_01_31_BonitaCanyon1',
  'data/2023_05_10_BonitaCanyon1',
  'data/2023_10_31_BonitaCanyon1',
  'data/2024_04_11_BonitaCanyon1',
  'data/2024_07_11_BonitaCanyon1',
  'data/2024_09_11_BonitaCanyon1',
  'data/2024_10_18_BonitaCanyon1',
  'data/2024_11_26_BonitaCanyon1',
]
# this could be improved by randomly sampling images from the full dataset so that they are uniformly distributed across time.
validation_data_paths = [
  'data/2024_02_02_BonitaCanyon1',
  'data/2025_01_15_BonitaCanyon1',
]

data_shuffle_buffer_size = 512 #smaller if 'Shuffle buffer filled.' error occurs

#training (go to training/model.py to configure the actual model structure)
image_size = (384, 216) # downsizes the images appropriately while maintaining aspect ratio
batch_size = 32
epochs = 10
