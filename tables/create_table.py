import pandas as pd
import sqlite3

# Path to your CSV file
csv_file_path = 'metadata.csv'

# Path to your SQLite database
sqlite_db_path = 'isic_data.db'

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Select only the required columns
# df_filtered = df[['isic_id', 'age_approx', 'benign_malignant', 'diagnosis', 'sex']]
df_filtered = df[['isic_id','attribution','copyright_license','acquisition_day','age_approx',
                  'anatom_site_general','benign_malignant','clin_size_long_diam_mm',
                  'concomitant_biopsy','dermoscopic_type','diagnosis','diagnosis_confirm_type',
                  'family_hx_mm','fitzpatrick_skin_type','image_type','lesion_id','mel_class',
                  'mel_mitotic_index','mel_thick_mm','mel_type','mel_ulcer','melanocytic',
                  'nevus_type','patient_id','personal_hx_mm','sex']]

# Connect to the SQLite database
conn = sqlite3.connect(sqlite_db_path)

# Define a function to chunk the DataFrame
def df_chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

# Specify the chunk size
chunk_size = 300  # Adjust based on your needs and testing

# Insert data in chunks
for chunk in df_chunker(df_filtered, chunk_size):
    chunk.to_sql('ISIC_DATA', conn, if_exists='append', index=False, method=None)

# Close the database connection
conn.close()
