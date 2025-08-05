from data.image_processing import create_baseline_image
import config

for folder_path in config.training_data_paths:
  print(folder_path)
  create_baseline_image(folder_path, True)
  create_baseline_image(folder_path, False)


for folder_path in config.validation_data_paths:
  print(folder_path)
  create_baseline_image(folder_path, True)
  create_baseline_image(folder_path, False)

# create_baseline_image('data/2022_12_16_BonitaCanyon2', True)
# create_baseline_image('data/2022_12_16_BonitaCanyon2', False)