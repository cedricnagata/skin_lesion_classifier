import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class MultiInputDataGenerator(Sequence):
    def __init__(self, image_paths, tabular_data, labels, batch_size=32, img_size=(224, 224), shuffle=True):
        self.image_paths = image_paths
        self.tabular_data = tabular_data
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_paths = [self.image_paths[k] for k in indexes]
        batch_tabular_data = np.array([self.tabular_data[k] for k in indexes])
        batch_labels = np.array([self.labels[k] for k in indexes])
        
        X_img = np.array([img_to_array(load_img(img_path, target_size=self.img_size)) for img_path in batch_image_paths]) / 255.0
        return (X_img, batch_tabular_data), batch_labels

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
