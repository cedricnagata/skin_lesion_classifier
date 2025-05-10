import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import logging
import pandas as pd

# Define base data directory
DATA_DIR = os.getenv("DATA_DIR")
TF_RECORDS_DIR = os.path.join(DATA_DIR, "tf-records")
MODEL_DIR = os.path.join(DATA_DIR, "models")

# Constants
IMG_HEIGHT, IMG_WIDTH = 450, 450
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

def parse_tfrecord(example_proto):
    """Parse TFRecord and return image and labels."""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'binary_label': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis_label': tf.io.FixedLenFeature([], tf.int64),
        'binary_weight': tf.io.FixedLenFeature([], tf.float32),
        'diagnosis_weight': tf.io.FixedLenFeature([], tf.float32),
    }
    
    features = tf.io.parse_single_example(example_proto, feature_description)
    
    # Decode and preprocess image
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    
    # Get labels and weights
    binary_label = features['binary_label']
    diagnosis_label = features['diagnosis_label']
    binary_weight = features['binary_weight']
    diagnosis_weight = features['diagnosis_weight']
    
    return image, (binary_label, diagnosis_label), (binary_weight, diagnosis_weight)

def create_dataset(tfrecord_path, batch_size, is_training=True):
    """Create a TensorFlow dataset from TFRecord file."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_model(num_diagnosis_classes):
    """Create a model with two output heads for binary and diagnosis classification."""
    # Base model (EfficientNetB0)
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Create the model
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    
    # Binary classification head
    binary_output = layers.Dense(1, activation='sigmoid', name='binary_output')(x)
    
    # Diagnosis classification head
    diagnosis_output = layers.Dense(num_diagnosis_classes, activation='softmax', name='diagnosis_output')(x)
    
    model = models.Model(inputs=inputs, outputs=[binary_output, diagnosis_output])
    
    return model

def train_model():
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Create datasets
    train_dataset = create_dataset(
        os.path.join(TF_RECORDS_DIR, 'train.tfrecord'),
        BATCH_SIZE,
        is_training=True
    )
    val_dataset = create_dataset(
        os.path.join(TF_RECORDS_DIR, 'val.tfrecord'),
        BATCH_SIZE,
        is_training=False
    )
    
    # Get number of diagnosis classes from metadata
    metadata_path = os.path.join(DATA_DIR, "metadata", "cleaned.csv")
    num_diagnosis_classes = len(pd.read_csv(metadata_path)['diagnosis_3'].unique())
    
    # Create and compile model
    model = create_model(num_diagnosis_classes)
    
    # Define loss functions and metrics
    losses = {
        'binary_output': tf.keras.losses.BinaryCrossentropy(),
        'diagnosis_output': tf.keras.losses.SparseCategoricalCrossentropy()
    }
    
    metrics = {
        'binary_output': ['accuracy', tf.keras.metrics.AUC()],
        'diagnosis_output': ['accuracy']
    }
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=losses,
        metrics=metrics
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        steps_per_epoch=1000,  # Adjust based on your dataset size
        validation_steps=100    # Adjust based on your validation set size
    )
    
    # Save the final model
    model.save(os.path.join(MODEL_DIR, 'final_model.h5'))
    
    # Save training history
    np.save(os.path.join(MODEL_DIR, 'training_history.npy'), history.history)

if __name__ == "__main__":
    train_model() 