# converts species code to numberic value for training (0 is false trigger and -1 is error and do not need to be included in the map)
label_map = {
  'CAFA': 1,
  'PRLO': 2,
  'HOSA': 3,
  'SCIU': 4,
  'BIRD': 5,
}


#training (go to training/model.py to configure the actual model structure)
image_size = (384, 216) # downsizes the images appropriately while maintaining aspect ratio
batch_size = 32
epochs = 10
