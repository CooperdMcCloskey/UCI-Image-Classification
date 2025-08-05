import numpy as np
from tensorflow.keras.callbacks import Callback # type: ignore


class PrintPredictionsCallback(Callback):
  def __init__(self, val_dataset, max_batches=1):
    super().__init__()
    self.val_dataset = val_dataset
    self.max_batches = max_batches  # How many batches to predict on (to limit print size)

  def on_epoch_end(self, epoch, logs=None):
    preds_list = []
    labels_list = []
    batches_processed = 0

    for batch in self.val_dataset:
        images, labels = batch
        preds = self.model.predict(images, verbose=0)
        
        preds_list.append(preds)
        labels_list.append(labels.numpy())
        
        batches_processed += 1
        if batches_processed >= self.max_batches:
            break

    preds_arr = np.concatenate(preds_list, axis=0).flatten()
    labels_arr = np.concatenate(labels_list, axis=0).flatten()
    rounded_preds = np.round(preds_arr).astype(int)

    # Limit output to first n elements
    limit = 24
    print(f"\nEpoch {epoch + 1} validation predictions (first {limit}):")
    print("Raw outputs:     ", preds_arr[:limit])
    print("Rounded outputs: ", rounded_preds[:limit])
    print("True labels:     ", labels_arr[:limit])