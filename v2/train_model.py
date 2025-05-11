import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, applications
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Creating dataset from {tfrecord_path}")
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    
    if is_training:
        dataset = dataset.shuffle(1000)
        dataset = dataset.repeat()
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Print dataset info
    for images, (binary_labels, diagnosis_labels), (binary_weights, diagnosis_weights) in dataset.take(1):
        logging.info(f"Image shape: {images.shape}")
        logging.info(f"Binary labels shape: {binary_labels.shape}")
        logging.info(f"Diagnosis labels shape: {diagnosis_labels.shape}")
        logging.info(f"Binary weights shape: {binary_weights.shape}")
        logging.info(f"Diagnosis weights shape: {diagnosis_weights.shape}")
        logging.info(f"Sample binary label: {binary_labels[0]}")
        logging.info(f"Sample diagnosis label: {diagnosis_labels[0]}")
        logging.info(f"Sample binary weight: {binary_weights[0]}")
        logging.info(f"Sample diagnosis weight: {diagnosis_weights[0]}")
    
    return dataset

def create_model(num_diagnosis_classes):
    """Create a model with two output heads for binary and diagnosis classification."""
    logging.info(f"Creating model with {num_diagnosis_classes} diagnosis classes")
    
    # Base model (EfficientNetB0)
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    logging.info("Base model (EfficientNetB0) is frozen")
    
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
    
    # Print model summary
    model.summary(print_fn=logging.info)
    
    return model

class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name='weighted_binary_crossentropy'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred, sample_weight=None):
        # Ensure y_true has the same shape as y_pred
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        
        if sample_weight is None:
            return tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.keras.losses.binary_crossentropy(y_true, y_pred) * sample_weight

class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name='weighted_sparse_categorical_crossentropy'):
        super().__init__(name=name)
    
    def call(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * sample_weight

class DebugCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logging.info(f"\nStarting epoch {epoch + 1}/{EPOCHS}")
    
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1} completed:")
        logging.info(f"  Binary accuracy: {logs['binary_output_accuracy']:.4f}")
        logging.info(f"  Binary AUC: {logs['binary_output_auc']:.4f}")
        logging.info(f"  Diagnosis accuracy: {logs['diagnosis_output_accuracy']:.4f}")
        logging.info(f"  Total loss: {logs['loss']:.4f}")
        logging.info(f"  Binary loss: {logs['binary_output_loss']:.4f}")
        logging.info(f"  Diagnosis loss: {logs['diagnosis_output_loss']:.4f}")
        logging.info(f"  Validation loss: {logs['val_loss']:.4f}")

def train_model():
    logging.info("Starting model training process")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    logging.info(f"Model directory: {MODEL_DIR}")
    
    # Create datasets
    logging.info("Creating training dataset...")
    train_dataset = create_dataset(
        os.path.join(TF_RECORDS_DIR, 'train.tfrecord'),
        BATCH_SIZE,
        is_training=True
    )
    
    logging.info("Creating validation dataset...")
    val_dataset = create_dataset(
        os.path.join(TF_RECORDS_DIR, 'val.tfrecord'),
        BATCH_SIZE,
        is_training=False
    )
    
    # Get number of diagnosis classes from metadata
    metadata_path = os.path.join(DATA_DIR, "metadata", "cleaned.csv")
    num_diagnosis_classes = len(pd.read_csv(metadata_path)['diagnosis_3'].unique())
    logging.info(f"Number of diagnosis classes: {num_diagnosis_classes}")
    
    # Create and compile model
    model = create_model(num_diagnosis_classes)
    
    # Define loss functions and metrics
    losses = {
        'binary_output': WeightedBinaryCrossentropy(),
        'diagnosis_output': WeightedSparseCategoricalCrossentropy()
    }
    
    metrics = {
        'binary_output': ['accuracy', tf.keras.metrics.AUC()],
        'diagnosis_output': ['accuracy']
    }
    
    # Compile model
    logging.info("Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss=losses,
        metrics=metrics
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.keras'),
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
        ),
        DebugCallback()
    ]
    
    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )
    
    # Save the final model
    logging.info("Saving final model...")
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    
    # Save training history
    logging.info("Saving training history...")
    np.save(os.path.join(MODEL_DIR, 'training_history.npy'), history.history)
    
    logging.info("Training completed successfully!")

if __name__ == "__main__":
    train_model() 