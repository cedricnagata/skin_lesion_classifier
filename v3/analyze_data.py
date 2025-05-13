import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_image_sizes(image_dir):
    """Analyze the distribution of image sizes in the dataset."""
    sizes = defaultdict(int)
    total_images = 0
    
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            try:
                with Image.open(os.path.join(image_dir, filename)) as img:
                    width, height = img.size
                    size_key = f"{width}x{height}"
                    sizes[size_key] += 1
                    total_images += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return sizes, total_images

def analyze_class_distribution(csv_path):
    """Analyze the distribution of classes in the diagnosis_1 column."""
    df = pd.read_csv(csv_path)
    class_distribution = df['diagnosis_1'].value_counts()
    return class_distribution

def main():
    DATA_DIR = os.getenv("DATA_DIR")

    # Define paths
    image_dir = os.path.join(DATA_DIR, "images", "raw")
    csv_path = os.path.join(DATA_DIR, "metadata", "raw.csv")
    
    print("Analyzing image sizes...")
    sizes, total_images = analyze_image_sizes(image_dir)
    print(f"\nTotal images processed: {total_images}")
    print("\nImage size distribution:")
    for size, count in sizes.items():
        print(f"{size}: {count} images ({(count/total_images)*100:.2f}%)")
    
    print("\nAnalyzing class distribution...")
    class_distribution = analyze_class_distribution(csv_path)
    print("\nClass distribution:")
    for class_name, count in class_distribution.items():
        print(f"{class_name}: {count} images ({(count/class_distribution.sum())*100:.2f}%)")

if __name__ == "__main__":
    main() 