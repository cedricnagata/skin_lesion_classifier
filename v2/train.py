import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision


# Load and Prepare Datasets
def parse_tfrecord(example_proto):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'diagnosis_label': tf.io.FixedLenFeature([], tf.int64),
    }
    features = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    return image, features['diagnosis_label']

def create_datasets(tf_records_dir, metadata_path, batch_size, shuffle_size):
    # Training dataset
    train_dataset = tf.data.TFRecordDataset(
        os.path.join(tf_records_dir, 'train.tfrecord'),
        num_parallel_reads=tf.data.AUTOTUNE
    ).map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    ).cache().shuffle(
        shuffle_size
    ).batch(
        batch_size
    ).prefetch(
        tf.data.AUTOTUNE
    )

    # Validation dataset
    val_dataset = tf.data.TFRecordDataset(
        os.path.join(tf_records_dir, 'val.tfrecord'),
        num_parallel_reads=tf.data.AUTOTUNE
    ).map(
        parse_tfrecord,
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(
        batch_size
    ).prefetch(
        tf.data.AUTOTUNE
    )

    df = pd.read_csv(metadata_path)
    num_samples = len(df)

    return train_dataset, val_dataset, num_samples

def build_model(img_height=450, img_width=450, base_trainable=False):
    mixed_precision.set_global_policy('mixed_float16')

    inputs = layers.Input(shape=(img_height, img_width, 3), name="input_image")
    x = layers.Rescaling(1.0/255)(inputs)

    base_model = tf.keras.applications.EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3)
    )
    base_model.trainable = base_trainable
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, outputs, name="EfficientNetV2L_skin_lesion")

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train_model(model, train_dataset, val_dataset, num_samples, batch_size, epochs):
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join('/models/base/', 'best_model_base.keras'),
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

    # Train Model with mixed precision
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )    

    return model, history

def fine_tune_model(
        model_or_path, 
        train_dataset, 
        val_dataset, 
        num_samples, 
        batch_size, 
        epochs, 
        base_lr=1e-5, 
        base_trainable=True
    ):
    """
    Fine-tune a previously trained model by unfreezing the base model and training with a lower learning rate.
    Accepts either a model object or a path to a saved model.
    """
    # Load the model if a path is given
    if isinstance(model_or_path, str):
        model = tf.keras.models.load_model(model_or_path)
    else:
        model = model_or_path

    # Unfreeze the base model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and 'efficientnet' in layer.name.lower():
            base_model = layer
            break
    if base_model is not None:
        base_model.trainable = base_trainable
    else:
        print("Warning: Could not find base model to unfreeze.")

    # Re-compile with a lower learning rate
    optimizer = tf.keras.optimizers.AdamW(learning_rate=base_lr, weight_decay=1e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    # Define callbacks (shorter patience for fine-tuning)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join('/models/tuned/', 'best_model_tuned.keras'),
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            min_lr=1e-7
        )
    ]

    steps_per_epoch = num_samples // batch_size

    # Fine-tune
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    return model, history
