from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(dataset):
  # Extract labels from your training dataset
  all_labels = []
  for _, labels in dataset:
      all_labels.extend(labels.numpy().flatten())  # flatten in case shape is (batch_size, 1)

  all_labels = np.array(all_labels)

  # Compute weights
  class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels), y=all_labels)

  # Format for Keras
  class_weight_dict = {i: w for i, w in enumerate(class_weights)}

  print("Computed class weights:", class_weight_dict)
  
  return class_weight_dict