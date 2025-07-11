from data import data_import
import config

for folder_path in config.training_data_paths:
  print(folder_path)
  data_import.createKey(folder_path)

for folder_path in config.validation_data_paths:
  print(folder_path)
  data_import.createKey(folder_path, True)