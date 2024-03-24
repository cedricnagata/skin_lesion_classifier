import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from glob import glob
import os

def load_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def get_isic_ids_from_folders(folders):
    ids = []
    for folder in folders:
        image_files = glob(os.path.join(folder, '*.jpg'))
        ids.extend([os.path.splitext(os.path.basename(image))[0] for image in image_files])
    return ids

def preprocess_features(df, preprocessor=None):
    features = df[['sex', 'age_approx', 'diagnosis']]
    
    if preprocessor is None:
        # Define preprocessing for categorical and numerical columns
        preprocessor = ColumnTransformer(transformers=[
            ('num', StandardScaler(), ['age_approx']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['sex', 'diagnosis']),
        ])
        # Fitting the preprocessor
        preprocessor.fit(features)
    
    # Applying the transformations
    features_preprocessed = preprocessor.transform(features)
    return features_preprocessed, preprocessor

def prepare_data_splits(metadata_path, image_folder_paths, preprocessor=None):
    metadata = load_metadata(metadata_path)
    data_splits = {}

    for split_name, folders in image_folder_paths.items():
        # Initialize an empty list for the image paths of the current split
        image_paths = []
        # Reset ids for each split to avoid carryover
        all_ids = []
        
        for folder in folders:
            # Collect ids for images in the current folder
            ids = get_isic_ids_from_folders([folder])
            all_ids.extend(ids)  # Extend the combined list of IDs
            # Add the correct path for each id
            image_paths.extend([os.path.join(folder, f"{id}.jpg") for id in ids])

        # Filter metadata to include only those IDs collected from folders
        split_metadata = metadata[metadata['isic_id'].isin(all_ids)]
        
        labels = np.where(split_metadata['benign_malignant'] == 'benign', 0, 1)
        features_preprocessed, preprocessor = preprocess_features(split_metadata, preprocessor)
        
        data_splits[split_name] = {
            'features_preprocessed': features_preprocessed,
            'labels': labels,
            'image_paths': image_paths  # Use the correctly constructed list of image paths
        }

    return data_splits, preprocessor
