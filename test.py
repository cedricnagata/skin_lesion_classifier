import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import tensorflow as tf

# Load the updated metadata with encoded labels and sample weights
metadata_file_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/updated_ground_truth_with_weights.csv'
metadata = pd.read_csv(metadata_file_path)

# Define image directory
image_folder_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/images/processed'

# Data normalization
test_datagen = ImageDataGenerator(rescale=1./255)

# Custom data generator for testing
class TestDataGenerator(Sequence):
    def __init__(self, dataframe, directory, x_col, y_col, target_size=(224, 224), batch_size=32, class_mode='multi_output', shuffle=False, seed=None):
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        batch_x = np.array([self.load_image(file_path) for file_path in batch_df[self.x_col]])
        batch_y = [batch_df[col].values for col in self.y_col]
        return batch_x, tuple(batch_y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_image(self, file_path):
        img = tf.keras.preprocessing.image.load_img(os.path.join(self.directory, file_path), target_size=self.target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = test_datagen.standardize(img)
        return img

# Create test data generator
test_generator = TestDataGenerator(
    dataframe=metadata,
    directory=image_folder_path,
    x_col='filename',
    y_col=['diagnosis_encoded', 'benign_malignant_encoded'],
    target_size=(224, 224),
    batch_size=32,
    class_mode='multi_output')

# Load the trained model
model = load_model('skin_lesion_classifier_final.keras')

# Evaluate the model on the test data
results = model.evaluate(test_generator, steps=len(test_generator))
print("Test Results - Loss: {:.4f}, Diagnosis Accuracy: {:.4f}, Benign/Malignant Accuracy: {:.4f}".format(
    results[0], results[1], results[2]))
