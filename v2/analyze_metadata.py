import pandas as pd
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def analyze_metadata(input_path, output_path):
    # Read the metadata
    logging.info(f"Reading metadata from {input_path}")
    df = pd.read_csv(input_path)
    logging.info(f"Original metadata shape: {df.shape}")

    # Check for missing values
    missing_values = df[['benign_malignant', 'diagnosis_3']].isnull().sum()
    logging.info("\nMissing values in key columns:")
    logging.info(missing_values)

    # Show unique values and their counts
    logging.info("\nUnique values in benign_malignant:")
    logging.info(df['benign_malignant'].value_counts())
    
    logging.info("\nUnique values in diagnosis_3:")
    logging.info(df['diagnosis_3'].value_counts())

    # Create new dataframe with only required columns
    clean_df = df[['benign_malignant', 'diagnosis_3']].copy()
    
    # Remove rows with missing values
    clean_df = clean_df.dropna()
    logging.info(f"\nCleaned metadata shape: {clean_df.shape}")
    logging.info(f"Removed {df.shape[0] - clean_df.shape[0]} rows with missing values")

    # Save the cleaned metadata
    clean_df.to_csv(output_path, index=False)
    logging.info(f"Saved cleaned metadata to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and clean metadata file")
    parser.add_argument('--metadata_path', required=True, help='Path to input metadata CSV file')
    parser.add_argument('--output_path', required=True, help='Path to save cleaned metadata CSV file')
    args = parser.parse_args()

    analyze_metadata(args.input, args.output)

if __name__ == "__main__":
    main() 