import os
import pandas as pd
import shutil
from pathlib import Path

def organize_images(metadata_path, image_dir, output_dir):
    """Organize resized images into benign and malignant folders with balanced classes."""
    # Create output directories
    benign_dir = os.path.join(output_dir, 'benign')
    malignant_dir = os.path.join(output_dir, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(metadata_path)
    
    # Filter out Indeterminate class and separate benign/malignant
    df = df[df['diagnosis_1'] != 'Indeterminate']
    benign_cases = df[df['diagnosis_1'] == 'Benign']
    malignant_cases = df[df['diagnosis_1'] == 'Malignant']
    
    # Sample benign cases to match malignant count
    if len(benign_cases) > len(malignant_cases):
        benign_cases = benign_cases.sample(n=len(malignant_cases), random_state=42)
    
    # Copy images to their respective directories
    for idx, row in benign_cases.iterrows():
        isic_id = row['isic_id']
        src_path = os.path.join(image_dir, f"{isic_id}.jpg")
        dst_path = os.path.join(benign_dir, f"{isic_id}.jpg")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    for idx, row in malignant_cases.iterrows():
        isic_id = row['isic_id']
        src_path = os.path.join(image_dir, f"{isic_id}.jpg")
        dst_path = os.path.join(malignant_dir, f"{isic_id}.jpg")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
    
    # Print final counts
    benign_count = len([f for f in os.listdir(benign_dir) if f.endswith('.jpg')])
    malignant_count = len([f for f in os.listdir(malignant_dir) if f.endswith('.jpg')])
    print(f"\nFinal dataset statistics:")
    print(f"Benign images: {benign_count}")
    print(f"Malignant images: {malignant_count}")

def main():
    DATA_DIR = "data/2019"
        
    # Define paths
    metadata_path = os.path.join(DATA_DIR, "metadata.csv")
    image_dir = os.path.join(DATA_DIR, "images", "raw_images")
    output_dir = os.path.join(DATA_DIR, "images", "organized_images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Organize images
    organize_images(metadata_path, image_dir, output_dir)

if __name__ == "__main__":
    main() 