import pandas as pd
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# Load the updated metadata with encoded labels and sample weights
metadata_file_path = 'C:/Users/thece/Documents/Software Projects/DERM DX/data/updated_ground_truth_with_weights.csv'
metadata = pd.read_csv(metadata_file_path)

# Ensure that the 'diagnosis_encoded' and 'benign_malignant_encoded' columns exist
assert 'diagnosis_encoded' in metadata.columns, "Missing 'diagnosis_encoded' column in metadata"
assert 'benign_malignant_encoded' in metadata.columns, "Missing 'benign_malignant_encoded' column in metadata"

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

# Display the model summary
model.summary()
