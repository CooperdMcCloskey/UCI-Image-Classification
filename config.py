# converts species code to numberic value for training (0 is false trigger and -1 is error and do not need to be included in the map)
label_map = {
  'UNK': -2,
  '#N/A': -2,
  'CAFA': 1,
  'PRLO': 2,
  'HOSA': 3,
  'SCIU': 4,
  'BIRD': 5,
  'RASP': 6,
  'OTOS': 7,
  'DIVI': 8,
  'NEOSP': 9,
  'LYRU': 10,
  'MEME': 11,
}

training_data_paths = [
  'data/2023_01_31_BonitaCanyon1',
  'data/2023_05_10_BonitaCanyon1',
  'data/2023_10_31_BonitaCanyon1',
  'data/2024_02_02_BonitaCanyon1',
  'data/2024_04_11_BonitaCanyon1',
  'data/2024_07_11_BonitaCanyon1',
  'data/2024_09_11_BonitaCanyon1',
  'data/2024_10_18_BonitaCanyon1',
  'data/2024_11_26_BonitaCanyon1',
]
validation_data_paths = [
  'data/2025_01_15_BonitaCanyon1',
]


#training (go to training/model.py to configure the actual model structure)
image_size = (384, 216) # downsizes the images appropriately while maintaining aspect ratio
batch_size = 32
epochs = 10
