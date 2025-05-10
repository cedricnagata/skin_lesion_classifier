import os
import numpy as np
from imblearn.over_sampling import SMOTE
import pandas as pd
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import logging

logging.basicConfig(level=logging.INFO)

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def encode_labels(metadata):
    metadata['diagnosis'] = metadata['diagnosis'].map({'nevus': 0, 'melanoma': 1, 'other': 2})
    metadata['benign_malignant'] = metadata['benign_malignant'].map({'benign': 0, 'malignant': 1})
    return metadata

def stratified_split(metadata, test_size=0.2, val_size=0.1):
    stratify_cols = metadata[['diagnosis', 'benign_malignant']]
    train_val_data, test_data = train_test_split(metadata, test_size=test_size, stratify=stratify_cols, random_state=42)
    train_data, val_data = train_test_split(train_val_data, test_size=val_size, stratify=train_val_data[['diagnosis', 'benign_malignant']], random_state=42)
    return train_data, val_data, test_data

def load_images(metadata, images_path, img_size=(224, 224)):
    images = []
    diagnoses = []
    benign_malignants = []
    for index, row in metadata.iterrows():
        try:
            img_path = os.path.join(images_path, row['filename'])
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            diagnoses.append(row['diagnosis'])
            benign_malignants.append(row['benign_malignant'])
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
    
    images = np.array(images)
    diagnoses = to_categorical(np.array(diagnoses), num_classes=3)
    benign_malignants = np.array(benign_malignants)
    
    return images, diagnoses, benign_malignants

def save_to_h5(file_path, images, diagnoses, benign_malignants):
    try:
        with h5py.File(file_path, 'w') as f:
            f.create_dataset('images', data=images)
            f.create_dataset('diagnoses', data=diagnoses)
            f.create_dataset('benign_malignants', data=benign_malignants)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")

def log_class_distribution(data, title):
    logging.info(f"{title} class distribution:")
    logging.info("Diagnosis:")
    logging.info(data['diagnosis'].value_counts(normalize=True))
    logging.info("Benign/Malignant:")
    logging.info(data['benign_malignant'].value_counts(normalize=True))

# Paths
metadata_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/metadata/GROUND_TRUTH.csv'
images_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/ISIC-images'
output_file = 'preprocessed_data.h5'

# Load and preprocess metadata
metadata = load_metadata(metadata_path)
metadata = encode_labels(metadata)
train_data, val_data, test_data = stratified_split(metadata)

# Log the number of samples in each split
logging.info(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}, Test samples: {len(test_data)}")

# Log class distribution in each split
log_class_distribution(train_data, "Train set")
log_class_distribution(val_data, "Validation set")
log_class_distribution(test_data, "Test set")

# Preload and save training data
train_images, train_diagnoses, train_benign_malignants = load_images(train_data, images_path)
save_to_h5('train_' + output_file, train_images, train_diagnoses, train_benign_malignants)

# Preload and save validation data
val_images, val_diagnoses, val_benign_malignants = load_images(val_data, images_path)
save_to_h5('val_' + output_file, val_images, val_diagnoses, val_benign_malignants)

# Preload and save test data
test_images, test_diagnoses, test_benign_malignants = load_images(test_data, images_path)
save_to_h5('test_' + output_file, test_images, test_diagnoses, test_benign_malignants)
