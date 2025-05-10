import os
import argparse
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

IMG_HEIGHT, IMG_WIDTH = 450, 450
SEED = 42
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Label maps
binary_map = {
    'benign': 0,
    'malignant': 1
}

diagnosis_map = {
    "Nevus": 0,
    "Melanoma, NOS": 1,
    "Pigmented benign keratosis": 2,
    "Basal cell carcinoma": 3,
    "Squamous cell carcinoma, NOS": 4,
    "Solar or actinic keratosis": 5,
    "Dermatofibroma": 6,
}

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(image_string, binary_label, diagnosis_label):
    feature = {
        'image': _bytes_feature(image_string),
        'binary_label': _int64_feature(binary_label),
        'diagnosis_label': _int64_feature(diagnosis_label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

def write_tfrecord(df_subset, image_dir, output_path):
    count = 0
    with tf.io.TFRecordWriter(output_path) as writer:
        for idx, row in df_subset.iterrows():
            image_path = os.path.join(image_dir, row['filename'])
            try:
                image_data = tf.io.read_file(image_path)
                example = serialize_example(
                    image_data.numpy(),
                    int(row['binary_label']),
                    int(row['diagnosis_label'])
                )
                writer.write(example)
                count += 1
            except Exception as e:
                logging.warning(f"Skipping {row['filename']}: {e}")
    logging.info(f"Wrote {count} records to {output_path}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Loading metadata...")
    df = pd.read_csv(args.metadata)
    logging.info(f"Loaded metadata with {len(df)} rows.")

    logging.info("Mapping string labels to integers...")
    df['binary_label'] = df['benign_malignant'].map(binary_map)
    df['diagnosis_label'] = df['diagnosis_3'].map(diagnosis_map)

    if df['binary_label'].isnull().any() or df['diagnosis_label'].isnull().any():
        raise ValueError("Unmapped labels found. Check your CSV and label maps.")

    df['stratify'] = df['binary_label'].astype(str) + "_" + df['diagnosis_label'].astype(str)

    logging.info("Splitting dataset into train/val/test...")
    train_val_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT,
        stratify=df['stratify'],
        random_state=SEED
    )

    val_rel_split = VAL_SPLIT / (1 - TEST_SPLIT)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_rel_split,
        stratify=train_val_df['stratify'],
        random_state=SEED
    )

    logging.info(f"Total: {len(df)} | Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    logging.info("Writing TFRecords...")
    write_tfrecord(train_df, args.image_dir, os.path.join(args.output_dir, 'train.tfrecord'))
    write_tfrecord(val_df, args.image_dir, os.path.join(args.output_dir, 'val.tfrecord'))
    write_tfrecord(test_df, args.image_dir, os.path.join(args.output_dir, 'test.tfrecord'))

    logging.info("TFRecord conversion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images and metadata to TFRecords with train/val/test split.")
    parser.add_argument('--image_dir', required=True, help='Path to image folder')
    parser.add_argument('--metadata', required=True, help='Path to CSV metadata file')
    parser.add_argument('--output_dir', required=True, help='Directory to save TFRecord files')
    args = parser.parse_args()

    main(args)