import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")
logging.info(f"Using data directory: {DATA_DIR}")

# Define all paths relative to DATA_DIR
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
METADATA_RAW_PATH = os.path.join(METADATA_DIR, "raw.csv")
METADATA_CLEANED_PATH = os.path.join(METADATA_DIR, "cleaned.csv")

def analyze_metadata(input_path, output_path):
    # Read the metadata
    df = pd.read_csv(input_path)
    logging.info(f"Original metadata shape: {df.shape}")

    # Use only 'isic_id' and 'diagnosis_1'
    clean_df = df[['isic_id', 'diagnosis_1']].copy()
    
    # Remove rows with missing values
    clean_df = clean_df.dropna()
    logging.info(f"After cleaning: {clean_df.shape[0]} rows")

    # Filter to only 'Benign' and 'Malignant'
    clean_df = clean_df[clean_df['diagnosis_1'].isin(['Benign', 'Malignant'])]
    logging.info(f"After filtering for Benign and Malignant: {clean_df.shape[0]} rows")

    # Balance the classes by downsampling 'Benign' to match 'Malignant'
    benign_df = clean_df[clean_df['diagnosis_1'] == 'Benign']
    malignant_df = clean_df[clean_df['diagnosis_1'] == 'Malignant']
    n_malignant = len(malignant_df)
    benign_sampled = benign_df.sample(n=n_malignant, random_state=42)
    balanced_df = pd.concat([benign_sampled, malignant_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    logging.info(f"After balancing: {balanced_df.shape[0]} rows (Benign: {n_malignant}, Malignant: {n_malignant})")

    # Show class distributions from final cleaned data
    logging.info("\nFinal Class Distribution:")
    class_counts = balanced_df['diagnosis_1'].value_counts()
    for name, count in class_counts.items():
        logging.info(f"{name:<10} {count:<6}")

    # Save the cleaned metadata
    balanced_df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned metadata to {output_path}")

if __name__ == "__main__":
    analyze_metadata(METADATA_RAW_PATH, METADATA_CLEANED_PATH)