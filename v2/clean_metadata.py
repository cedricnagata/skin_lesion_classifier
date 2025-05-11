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

    # Create new dataframe with only required columns
    clean_df = df[['isic_id', 'diagnosis_2']].copy()
    
    # Remove rows with missing values
    clean_df = clean_df.dropna()
    logging.info(f"After cleaning: {clean_df.shape[0]} rows")

    # Filter out classes with only one sample
    single_sample_classes = clean_df.groupby('diagnosis_2').filter(lambda x: len(x) < 100)['diagnosis_2'].unique()
    if len(single_sample_classes) > 0:
        clean_df = clean_df[~clean_df['diagnosis_2'].isin(single_sample_classes)]
        logging.info(f"After removing single-sample classes: {clean_df.shape[0]} rows")

    # Show class distributions from final cleaned data
    logging.info("\nFinal Class Distribution:")
    logging.info("\nDiagnosis Distribution (diagnosis_2):")
    logging.info(clean_df['diagnosis_2'].value_counts())

    # Save the cleaned metadata
    clean_df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned metadata to {output_path}")

if __name__ == "__main__":
    analyze_metadata(METADATA_RAW_PATH, METADATA_CLEANED_PATH)