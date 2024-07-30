import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
import tensorflow as tf
import numpy as np
import os

# Load the updated metadata with encoded labels and sample weights
metadata_file_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/updated_ground_truth_with_weights.csv'
metadata = pd.read_csv(metadata_file_path)

# Define image directory
image_folder_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/images/processed'

# Data augmentation and normalization
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

# Custom data generator to include sample weights
class DataGeneratorWithWeights(Sequence):
    def __init__(self, dataframe, directory, x_col, y_col, sample_weights_col, datagen, target_size=(224, 224), batch_size=32, class_mode='multi_output', shuffle=True, seed=None):
        self.datagen = datagen
        self.dataframe = dataframe
        self.directory = directory
        self.x_col = x_col
        self.y_col = y_col
        self.sample_weights_col = sample_weights_col
        self.target_size = target_size
        self.batch_size = batch_size
        self.class_mode = class_mode
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indexes]
        batch_x = np.array([self.load_image(file_path) for file_path in batch_df[self.x_col]])
        batch_y = [batch_df[col].values for col in self.y_col]
        sample_weights = batch_df[self.sample_weights_col].values
        return batch_x, tuple(batch_y), sample_weights

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_image(self, file_path):
        img = tf.keras.preprocessing.image.load_img(os.path.join(self.directory, file_path), target_size=self.target_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = self.datagen.random_transform(img)
        img = self.datagen.standardize(img)
        return img

# Split data into training, validation, and test sets
train_df, temp_df = train_test_split(metadata, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create data generators
train_generator_with_weights = DataGeneratorWithWeights(
    dataframe=train_df,
    directory=image_folder_path,
    x_col='filename',
    y_col=['diagnosis_encoded', 'benign_malignant_encoded'],
    sample_weights_col='sample_weights',
    datagen=train_datagen,
    target_size=(224, 224),
    batch_size=32,
    class_mode='multi_output')

val_generator_with_weights = DataGeneratorWithWeights(
    dataframe=val_df,
    directory=image_folder_path,
    x_col='filename',
    y_col=['diagnosis_encoded', 'benign_malignant_encoded'],
    sample_weights_col='sample_weights',
    datagen=val_datagen,
    target_size=(224, 224),
    batch_size=32,
    class_mode='multi_output')

# Load pretrained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layers
num_diagnosis_classes = metadata['diagnosis_encoded'].nunique()
diagnosis_output = Dense(num_diagnosis_classes, activation='softmax', name='diagnosis_output')(x)
bm_output = Dense(2, activation='softmax', name='bm_output')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=[diagnosis_output, bm_output])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss={'diagnosis_output': 'sparse_categorical_crossentropy', 'bm_output': 'sparse_categorical_crossentropy'},
              metrics={'diagnosis_output': 'accuracy', 'bm_output': 'accuracy'})

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
checkpoint = ModelCheckpoint('skin_lesion_classifier.keras', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_generator_with_weights,
    steps_per_epoch=len(train_generator_with_weights),
    validation_data=val_generator_with_weights,
    validation_steps=len(val_generator_with_weights),
    epochs=20,
    callbacks=[early_stopping, reduce_lr, checkpoint])

# Save the final model
model.save('skin_lesion_classifier_final.keras')