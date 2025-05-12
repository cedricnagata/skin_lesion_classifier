import os
import logging
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")
logging.info(f"Using data directory: {DATA_DIR}")

# Define all paths relative to DATA_DIR
IMAGE_PROCESSED_DIR = os.path.join(DATA_DIR, "images", "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
TF_RECORDS_DIR = os.path.join(DATA_DIR, "tf-records")

SEED = 42
VAL_SPLIT = 0.20

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_string, diagnosis_label):
    feature = {
        "image": _bytes_feature(image_string),
        "diagnosis_label": _int64_feature(diagnosis_label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def write_tfrecord(df_subset, image_dir, output_path):
    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for idx, row in df_subset.iterrows():
            image_path = os.path.join(image_dir, f"{row['isic_id']}.jpg")
            try:
                image_data = tf.io.read_file(image_path)
                example = serialize_example(
                    image_data.numpy(),
                    int(row["diagnosis_label"]),
                )
                writer.write(example)
                count += 1
            except Exception as e:
                logging.warning(f"Skipping {row['isic_id']}: {e}")
    logging.info(f"Created {output_path} with {count} records")

def create_tf_records(df, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Map 'Benign' to 0 and 'Malignant' to 1
    label_map = {'Benign': 0, 'Malignant': 1}
    df = df[df['diagnosis_1'].isin(label_map.keys())].copy()
    df["diagnosis_label"] = df["diagnosis_1"].map(label_map)

    # Split dataset into train and val only
    train_df, val_df = train_test_split(
        df, 
        test_size=VAL_SPLIT, 
        stratify=df["diagnosis_label"], 
        random_state=SEED
    )
    logging.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}")

    # Write TFRecords
    write_tfrecord(train_df, image_dir, os.path.join(output_dir, "train.tfrecord"))
    write_tfrecord(val_df, image_dir, os.path.join(output_dir, "val.tfrecord"))

if __name__ == "__main__":
    cleaned_df = pd.read_csv(os.path.join(METADATA_DIR, "cleaned.csv"))
    create_tf_records(cleaned_df, IMAGE_PROCESSED_DIR, TF_RECORDS_DIR)
