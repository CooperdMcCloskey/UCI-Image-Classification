import matplotlib.pyplot as plt

from data.data_import import createDataset
from config import training_data_paths

dataset = createDataset(paths=training_data_paths, validation=False)

for images, labels in dataset.take(1):  
  for i in range(min(8, images.shape[0])):  
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy())  
    plt.title(f"Label: {labels[i].numpy()}")
    plt.axis("off")
  plt.tight_layout()
  plt.show()
