# important settings
classification_type = 'general' # use general if classifying based on false / true trigger; use per-species if trying to classify each species individually.
time = 'night' # training model for daytime or nighttime images (there are separate models for each)
is_day = time == 'day'


#training (go to training/model.py to configure the actual model structure)
image_size = (224, 224) # downsizes the images appropriately while maintaining aspect ratio
batch_size =24
epochs = 30

debugging = True
# converts species code to numberic value for training (0 is false trigger and -1 is error and do not need to be included in the map)
per_species_label_map = {
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

general_label_map = {
  'UNK': -2,
  '#N/A': -2,
  'VEHI': 1, # vehicle - counted as false trigger
  'BIKE': 1, # bike - counted as false trigger
  'HOSA': 1, # human - counted as false trigger
  'CAFA': 1, # coyote
  'LYRU': 1, # bobcat
  'RASP': 1, # rabbit
  'PRLO': 1, # raccoon
  'BIRD': 1, # bird
  'DIVI': 1, # opossum
  'MEME': 1, # skunk
  'SCIU': 1, # squirrel
  'OTOS': 1, # ground squirrel - grouped with squirrels
  'NEOSP': 1, # wood rat
}

per_species_class_count = 10 # 0-9 is 10 unique classes
general_class_count = 1 # either false trigger (0) or true (1)


label_map = {}
class_weights = {}
class_count = 0

if classification_type == 'general':
  label_map = general_label_map
  class_count = general_class_count
else:
  label_map = per_species_label_map
  class_count = per_species_class_count
	

training_data_paths = [
  'data/2023_01_31_BonitaCanyon1',
  'data/2023_05_10_BonitaCanyon1',
  'data/2023_10_31_BonitaCanyon1',
  'data/2024_04_11_BonitaCanyon1',
  'data/2024_07_11_BonitaCanyon1',
  'data/2024_09_11_BonitaCanyon1',
  'data/2024_10_18_BonitaCanyon1',
  'data/2024_11_26_BonitaCanyon1',
	'data/2022_11_09_BonitaCanyon2',
	'data/2022_12_16_BonitaCanyon2',
	'data/2023_06_21_BonitaCanyon2',
  'data/2024_07_11_BonitaCanyon2',
	'data/2024_09_11_BonitaCanyon2',
  'data/2024_10_18_BonitaCanyon2',
]
# this could be improved by randomly sampling images from the full dataset so that they are uniformly distributed across time.
validation_data_paths = [
  'data/2024_02_02_BonitaCanyon1',
  'data/2025_01_15_BonitaCanyon1',
  'data/2024_02_02_BonitaCanyon2',
	'data/2023_01_31_BonitaCanyon2',
]

data_shuffle_buffer_size = 1000 #smaller if 'Shuffle buffer filled.' error occurs

