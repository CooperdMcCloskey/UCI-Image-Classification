from collections import Counter
import csv
import config

def count_labels(paths):
   
  image_labels = []
  for path in paths:
    with open(path+'/key.csv', 'r') as file:
      csvreader = csv.reader(file)
      next(csvreader)
      for row in csvreader:
        image_labels.append(int(row[1]))

  label_counts = Counter(image_labels)

  for label, count in sorted(label_counts.items(), key=lambda x: int(x[0])):
      print(f"Label {label}: {count}")

print('Training Data -------------------')
count_labels(config.training_data_paths)

print('Validation Data -----------------')
count_labels(config.validation_data_paths)