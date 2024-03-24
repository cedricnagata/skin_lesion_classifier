import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Define paths
metadata_path = 'tables/metadata.csv'
images_base_path = 'images/processed'  # Assuming all images are initially here
output_base_path = 'images/training'  # Where to save the organized structure

# Load metadata
metadata = pd.read_csv(metadata_path)

# Create output directories
for category in ['benign', 'malignant']:
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_base_path, category, split), exist_ok=True)

def distribute_images(diagnosis_group, output_path):
    num_images = len(diagnosis_group)
    
    if num_images == 1:
        # Directly assign the single image to the training set.
        train = diagnosis_group
        val = []
        test = []
    elif num_images == 2:
        # Split two images between training and validation.
        train, val = train_test_split(diagnosis_group, test_size=0.5, random_state=42)
        test = []
    elif num_images == 3:
        # With three images, assign one to each set, prioritizing training and validation.
        train, temp = train_test_split(diagnosis_group, test_size=2/3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
    else:
        # Normal distribution for larger sets
        train, test_val = train_test_split(diagnosis_group, test_size=0.3, random_state=42)
        val, test = train_test_split(test_val, test_size=1/3, random_state=42)  # Achieves 20% validation, 10% test split

    # Function to handle the copying of files
    def copy_files(files, split_name):
        for file_name in files:
            src = os.path.join(images_base_path, file_name)
            dst = os.path.join(output_path, split_name, file_name)
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

    # Copy files to their designated directories
    copy_files(train, 'train')
    copy_files(val, 'val')
    copy_files(test, 'test')

# Iterate through benign and malignant categories
for category in metadata['benign_malignant'].unique():
    category_path = os.path.join(output_base_path, category)
    category_metadata = metadata[metadata['benign_malignant'] == category]

    # Iterate through each diagnosis in the category
    for diagnosis in category_metadata['diagnosis'].unique():
        diagnosis_metadata = category_metadata[category_metadata['diagnosis'] == diagnosis]
        # Construct the image file names (assuming .jpg extension)
        diagnosis_images = diagnosis_metadata['isic_id'].apply(lambda x: f"{x}.jpg").tolist()
        distribute_images(diagnosis_images, category_path)

print("Images have been organized.")