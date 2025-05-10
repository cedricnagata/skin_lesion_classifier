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
    missing_values = df[['diagnosis_1', 'diagnosis_3', 'isic_id']].isnull().sum()
    logging.info("\nMissing values in key columns:")
    logging.info(missing_values)

    # Create new dataframe with only required columns
    clean_df = df[['isic_id', 'diagnosis_1', 'diagnosis_3']].copy()
    
    # Remove rows with missing values
    clean_df = clean_df.dropna()
    logging.info(f"\nAfter removing missing values: {clean_df.shape[0]} rows")

    # Filter out 'Indeterminate' diagnoses
    clean_df = clean_df[clean_df['diagnosis_1'].isin(['Benign', 'Malignant'])]
    logging.info(f"After removing 'Indeterminate' diagnoses: {clean_df.shape[0]} rows")

    # Show all unique values and their counts
    logging.info("\nFinal unique values in diagnosis_1:")
    logging.info(clean_df['diagnosis_1'].value_counts())
    
    logging.info("\nFinal unique values in diagnosis_3:")
    logging.info(clean_df['diagnosis_3'].value_counts())

    # Save the cleaned metadata
    clean_df.to_csv(output_path, index=False)
    logging.info(f"\nSaved cleaned metadata to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and clean metadata file")
    parser.add_argument('--metadata_path', required=True, help='Path to input metadata CSV file')
    parser.add_argument('--output_path', required=True, help='Path to save cleaned metadata CSV file')
    args = parser.parse_args()

    analyze_metadata(args.metadata_path, args.output_path)

if __name__ == "__main__":
    main() 