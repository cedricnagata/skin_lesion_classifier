import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from sklearn.utils.class_weight import compute_class_weight


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
    ).cache(
        '/content/tf_cache'
    ).shuffle(
        shuffle_size
    ).batch(
        batch_size, drop_remainder=True
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
        batch_size, drop_remainder=True
    ).prefetch(
        tf.data.AUTOTUNE
    )

    df = pd.read_csv(metadata_path)
    num_diagnosis_classes = len(df['diagnosis_2'].unique())
    num_samples = len(df)

    # Calculate class weights
    y = df['diagnosis_2'].map({name: idx for idx, name in enumerate(sorted(df['diagnosis_2'].unique()))})
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    return train_dataset, val_dataset, num_diagnosis_classes, num_samples, class_weight_dict

def build_model(num_diagnosis_classes, img_height=1024, img_width=1024, base_trainable=False):
    mixed_precision.set_global_policy('mixed_float16')

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")

    inputs = layers.Input(shape=(img_height, img_width, 3), name="input_image")
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0/255)(x)

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
    outputs = layers.Dense(num_diagnosis_classes, activation="softmax", dtype="float32")(x)

    model = models.Model(inputs, outputs, name="EfficientNetV2L_skin_lesion")

    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)

    model.compile(
        optimizer=optimizer,
        loss=WeightedSparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")]
    )
    return model

def train_model(model, train_dataset, val_dataset, num_samples, model_dir, batch_size, epochs, class_weight):
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
        ),
        # Add TensorBoard callback
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs'),
            histogram_freq=1,
            profile_batch='500,520'
        )
    ]

    steps_per_epoch = num_samples // batch_size

    # Train Model with mixed precision
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weight
    )    

    return model, history

def fine_tune_model(
        model_path, 
        train_dataset, 
        val_dataset, 
        num_samples, 
        model_dir, 
        batch_size, 
        epochs, 
        class_weight, 
        base_lr=1e-5, 
        base_trainable=True
    ):
    """
    Fine-tune a previously trained model by unfreezing the base model and training with a lower learning rate.
    """
    # Load the model
    model = tf.keras.models.load_model(model_path, custom_objects={
        'WeightedSparseCategoricalCrossentropy': WeightedSparseCategoricalCrossentropy
    })

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
        loss=WeightedSparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy")]
    )

    # Define callbacks (shorter patience for fine-tuning)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'finetuned_model.keras'),
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
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(model_dir, 'logs_finetune'),
            histogram_freq=1,
            profile_batch='500,520'
        )
    ]

    steps_per_epoch = num_samples // batch_size

    # Fine-tune
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        class_weight=class_weight
    )

    return model, history
