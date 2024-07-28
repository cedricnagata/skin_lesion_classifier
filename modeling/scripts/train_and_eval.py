import numpy as np
from sklearn.utils import compute_class_weight
from multi_input_model import create_multi_input_model
from multi_input_data_generator import MultiInputDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import matplotlib.pyplot as plt

# Assuming data_preparation.py contains a function 'prepare_data'
# that returns a dictionary with preprocessed tabular data and labels for each split
from data_preparation import prepare_data_splits

# Paths to image directories and metadata file
metadata_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/metadata.csv'
image_folder_paths = {
    'train': ['C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/benign/train', 
              'C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/malignant/train'],
    'val': ['C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/benign/val', 
            'C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/malignant/val'],
    'test': ['C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/benign/test', 
             'C:/Users/thece/Documents/Software Projects/DERM DX/data/images/training/malignant/test']
}

data_splits, _ = prepare_data_splits(metadata_path, image_folder_paths)

# Ensure data integrity and correct dimensions
for split_name, split_data in data_splits.items():
    image_paths = split_data['image_paths']
    tabular_data = split_data['features_preprocessed']
    labels = split_data['labels']
    
    # Convert tabular data from sparse to dense if necessary
    if hasattr(tabular_data, "toarray"):
        tabular_data = tabular_data.toarray()
        data_splits[split_name]['features_preprocessed'] = tabular_data
    
    tabular_len = tabular_data.shape[0]
    print(f"{split_name} image paths: {len(image_paths)}, tabular data: {tabular_len}, labels: {len(labels)}")
    assert len(image_paths) == tabular_len == len(labels), f"Mismatch in dataset lengths for {split_name} split."

# Instantiate data generators
train_generator = MultiInputDataGenerator(
    image_paths=data_splits['train']['image_paths'],
    tabular_data=data_splits['train']['features_preprocessed'],
    labels=data_splits['train']['labels']
)

val_generator = MultiInputDataGenerator(
    image_paths=data_splits['val']['image_paths'],
    tabular_data=data_splits['val']['features_preprocessed'],
    labels=data_splits['val']['labels']
)

test_generator = MultiInputDataGenerator(
    image_paths=data_splits['test']['image_paths'],
    tabular_data=data_splits['test']['features_preprocessed'],
    labels=data_splits['test']['labels']
)

# Example custom callback for logging
class CustomLoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Ending Epoch {epoch}, Training Loss: {logs['loss']}, Validation Loss: {logs['val_loss']}")

# Initialize and compile your model here
model = create_multi_input_model()

_labels = data_splits['train']['labels']
class_weights = compute_class_weight('balanced', classes=np.unique(_labels), y=_labels)
class_weights_dict = dict(enumerate(class_weights))

# Display the model's architecture
model.summary()

# Define callbacks
checkpoint_cb = ModelCheckpoint("best_model.keras", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)
custom_logging_cb = CustomLoggingCallback()

# Proceed with model.fit() as you have it
history = model.fit(
    train_generator,  # Make sure train_generator is defined and loaded with your data
    validation_data=val_generator,  # Ensure val_generator is also ready
    epochs=3,  # Adjust as needed
    class_weight=class_weights_dict,
    callbacks=[checkpoint_cb, early_stopping_cb, custom_logging_cb],
    verbose=1  # Set verbose to 2 for one line per epoch, 1 for more detailed output
)

# Plotting training and validation loss and accuracy
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss / Accuracy')
plt.ylabel('Loss / Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# After training, evaluate your model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")