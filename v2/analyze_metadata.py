import pandas as pd
import argparse

def analyze_metadata(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    print("\nBenign vs Malignant Distribution:")
    print("-" * 30)
    benign_malignant_counts = df['benign_malignant'].value_counts()
    for category, count in benign_malignant_counts.items():
        print(f"{category}: {count} images")
    
    print("\nDiagnosis 3 Distribution:")
    print("-" * 30)
    diagnosis_counts = df['diagnosis_3'].value_counts()
    for diagnosis, count in diagnosis_counts.items():
        print(f"{diagnosis}: {count} images")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze metadata CSV file for benign/malignant and diagnosis distributions')
    parser.add_argument('csv_path', help='Path to the metadata CSV file')
    
    args = parser.parse_args()
    
    try:
        analyze_metadata(args.csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the CSV file at {args.csv_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}") 