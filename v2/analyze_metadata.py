import os
import pandas as pd

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")

# Define all paths relative to DATA_DIR
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
METADATA_RAW_PATH = os.path.join(METADATA_DIR, "raw.csv")
METADATA_CLEANED_PATH = os.path.join(METADATA_DIR, "cleaned.csv")

def analyze_metadata(input_path, output_path):
    # Create a list to store summary information
    summary_lines = []
    
    # Read the metadata
    summary_lines.append(f"Reading metadata from {input_path}")
    df = pd.read_csv(input_path)
    summary_lines.append(f"Original metadata shape: {df.shape}")

    # Check for missing values
    missing_values = df[['diagnosis_1', 'diagnosis_3', 'isic_id']].isnull().sum()
    summary_lines.append("\nMissing values in key columns:")
    summary_lines.append(str(missing_values))

    # Create new dataframe with only required columns
    clean_df = df[['isic_id', 'diagnosis_1', 'diagnosis_3']].copy()
    
    # Remove rows with missing values
    clean_df = clean_df.dropna()
    summary_lines.append(f"\nAfter removing missing values: {clean_df.shape[0]} rows")

    # Filter diagnosis_1 to only include 'Benign' and 'Malignant'
    clean_df = clean_df[clean_df['diagnosis_1'].isin(['Benign', 'Malignant'])]
    summary_lines.append(f"After removing 'Indeterminate' diagnoses: {clean_df.shape[0]} rows")

    # Show all unique values and their counts
    summary_lines.append("\nFinal unique values in diagnosis_1:")
    summary_lines.append(str(clean_df['diagnosis_1'].value_counts()))
    
    summary_lines.append("\nFinal unique values in diagnosis_3:")
    summary_lines.append(str(clean_df['diagnosis_3'].value_counts()))

    # Save the cleaned metadata
    clean_df.to_csv(output_path, index=False)
    summary_lines.append(f"\nSaved cleaned metadata to {output_path}")
    
    # Write summary to file
    summary_path = os.path.join(os.path.dirname(output_path), "summary.txt")
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))

if __name__ == "__main__":
    # Run metadata analysis and cleaning
    analyze_metadata(METADATA_RAW_PATH, METADATA_CLEANED_PATH)