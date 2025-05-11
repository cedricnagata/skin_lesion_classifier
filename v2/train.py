import os
import tensorflow as tf
import numpy as np
import pandas as pd


# Define Custom Loss Classes
class WeightedSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, name='weighted_sparse_categorical_crossentropy'):
        super().__init__(name=name)

    def call(self, y_true, y_pred, sample_weight=None):
        if sample_weight is None:
            return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred) * sample_weight
    

# Load and Prepare Datasets
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'diagnosis_label': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis_weight': tf.io.FixedLenFeature([], tf.float32),
    }

    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image, features['diagnosis_label'], features['diagnosis_weight']

def create_datasets(tf_records_dir, metadata_path, batch_size, shuffle_size):
    train_dataset = tf.data.TFRecordDataset(os.path.join(tf_records_dir, 'train.tfrecord'))
    train_dataset = train_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(shuffle_size)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.TFRecordDataset(os.path.join(tf_records_dir, 'val.tfrecord'))
    val_dataset = val_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    df = pd.read_csv(metadata_path)
    num_diagnosis_classes = len(df['diagnosis_2'].unique())
    num_samples = len(df)

    return train_dataset, val_dataset, num_diagnosis_classes, num_samples


def build_model(num_diagnosis_classes, img_height, img_width):
    # Input shape
    input_shape = (img_height, img_width, 3)

    # Load ResNet50 base (pretrained on ImageNet)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base_model.trainable = True  # Set to True for fine-tuning

    # Build custom model
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # Important for batch norm layers
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    # Output heads
    diagnosis_output = tf.keras.layers.Dense(num_diagnosis_classes, activation='softmax', name='diagnosis_output')(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=[diagnosis_output])

    # Define losses and metrics
    losses = {
        'diagnosis_output': WeightedSparseCategoricalCrossentropy()
    }
    metrics = {
        'diagnosis_output': ['accuracy']
    }

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        metrics=metrics
    )

    return model

def train_model(model, train_dataset, val_dataset, num_samples, model_dir, batch_size, epochs):
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'best_model.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        )
    ]

    steps_per_epoch = num_samples // batch_size

    # Train Model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )    

    return model, history
