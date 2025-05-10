import os
import logging
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")

# Define all paths relative to DATA_DIR
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMAGE_RAW_DIR = os.path.join(IMAGE_DIR, "raw")
IMAGE_PROCESSED_DIR = os.path.join(IMAGE_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
METADATA_RAW_PATH = os.path.join(METADATA_DIR, "raw.csv")
METADATA_CLEANED_PATH = os.path.join(METADATA_DIR, "cleaned.csv")
TF_RECORDS_DIR = os.path.join(DATA_DIR, "tf-records")

# Constants
IMG_HEIGHT, IMG_WIDTH = 450, 450 # Change as needed
SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def serialize_example(image_string, binary_label, diagnosis_label, binary_weight, diagnosis_weight):
    feature = {
        "image": _bytes_feature(image_string),
        "binary_label": _int64_feature(binary_label),
        "diagnosis_label": _int64_feature(diagnosis_label),
        "binary_weight": _float_feature(binary_weight),
        "diagnosis_weight": _float_feature(diagnosis_weight),
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
                    int(row["binary_label"]),
                    int(row["diagnosis_label"]),
                    float(row["binary_weight"]),
                    float(row["diagnosis_weight"]),
                )
                writer.write(example)
                count += 1
            except Exception as e:
                logging.warning(f"Skipping {row['isic_id']}: {e}")
    logging.info(f"Wrote {count} records to {output_path}")

def create_tf_records(df, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Dynamically create label maps from the CSV
    binary_classes = sorted(df["diagnosis_1"].dropna().unique())
    binary_map = {name: idx for idx, name in enumerate(binary_classes)}
    diagnosis_classes = sorted(df["diagnosis_3"].dropna().unique())
    diagnosis_map = {name: idx for idx, name in enumerate(diagnosis_classes)}
    logging.info(f"Binary label mapping: {binary_map}")
    logging.info(f"Diagnosis label mapping: {diagnosis_map}")

    # Map string labels to integers
    df["binary_label"] = df["diagnosis_1"].map(binary_map)
    df["diagnosis_label"] = df["diagnosis_3"].map(diagnosis_map)

    if df["binary_label"].isnull().any() or df["diagnosis_label"].isnull().any():
        raise ValueError("Unmapped labels found. Check your CSV and label maps.")

    # Compute class weights
    binary_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.sort(df['binary_label'].unique()),
        y=df['binary_label']
    )
    binary_class_weight = dict(zip(np.sort(df['binary_label'].unique()), binary_weights))

    diagnosis_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.sort(df['diagnosis_label'].unique()),
        y=df['diagnosis_label']
    )
    diagnosis_class_weight = dict(zip(np.sort(df['diagnosis_label'].unique()), diagnosis_weights))

    df['binary_weight'] = df['binary_label'].map(binary_class_weight)
    df['diagnosis_weight'] = df['diagnosis_label'].map(diagnosis_class_weight)

    df["stratify"] = (
        df["binary_label"].astype(str) + "_" + df["diagnosis_label"].astype(str)
    )

    # Split dataset
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, stratify=df["stratify"], random_state=SEED
    )

    val_rel_split = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_split,
        stratify=train_val_df["stratify"],
        random_state=SEED,
    )

    logging.info(
        f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}"
    )

    # Write TFRecords
    write_tfrecord(
        train_df, image_dir, os.path.join(output_dir, "train.tfrecord")
    )
    write_tfrecord(
        val_df, image_dir, os.path.join(output_dir, "val.tfrecord")
    )
    write_tfrecord(
        test_df, image_dir, os.path.join(output_dir, "test.tfrecord")
    )

    logging.info("TFRecord conversion completed successfully.")


if __name__ == "__main__":
    # Run TFRecord creation
    cleaned_df = pd.read_csv(os.path.join(METADATA_DIR, "cleaned.csv"))
    create_tf_records(cleaned_df, IMAGE_PROCESSED_DIR, TF_RECORDS_DIR)