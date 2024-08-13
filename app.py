from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import psutil  # For memory tracking
import gc

app = Flask(__name__)

# Define the labels
diagnosis_labels = ['nevus', 'melanoma', 'other']
benign_malignant_labels = ['benign', 'malignant']

def load_model_with_optimization(model_path):
    model = load_model(model_path, compile=False)

    return model

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32)
    img_array /= 255.0
    return img_array

def log_memory_usage(stage):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    peak_memory = memory_info.peak_wset if hasattr(memory_info, 'peak_wset') else memory_info.rss
    print(f"[{stage}] Peak memory usage: {peak_memory / (1024 ** 2):.2f} MiB")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save the file to a temporary location
    temp_file_path = '/tmp/temp.jpg'
    file.save(temp_file_path)

    # Preprocess the image
    img_array = preprocess_image(temp_file_path)
    log_memory_usage("After image preprocessing")

    # Load, predict, and unload diagnosis model
    diagnosis_model_path = 'diagnosis_model.h5'
    diagnosis_model = load_model_with_optimization(diagnosis_model_path)
    diagnosis_predictions = diagnosis_model.predict(img_array)
    del diagnosis_model  # Unload the model to free up memory
    gc.collect()
    log_memory_usage("After diagnosis model prediction")

    # Process diagnosis predictions
    diagnosis_pred = np.argmax(diagnosis_predictions, axis=1)[0]
    diagnosis_confidence = diagnosis_predictions[0][diagnosis_pred]

    # Load, predict, and unload benign/malignant model
    benign_malignant_model_path = 'benign_malignant_model.h5'
    benign_malignant_model = load_model_with_optimization(benign_malignant_model_path)
    benign_malignant_predictions = benign_malignant_model.predict(img_array)
    del benign_malignant_model  # Unload the model to free up memory
    gc.collect()
    log_memory_usage("After benign/malignant model prediction")

    # Process benign/malignant predictions
    benign_malignant_pred = int(benign_malignant_predictions[0] > 0.5)
    benign_malignant_confidence = benign_malignant_predictions[0][0] if benign_malignant_pred else 1 - benign_malignant_predictions[0][0]

    # Map predictions to labels
    diagnosis = diagnosis_labels[diagnosis_pred]
    benign_malignant = benign_malignant_labels[benign_malignant_pred]

    # Remove the temporary file
    os.remove(temp_file_path)
    log_memory_usage("After removing temporary file")

    # Return the predictions
    return jsonify({
        'diagnosis': diagnosis,
        'diagnosis_confidence': float(diagnosis_confidence),
        'benign_malignant': benign_malignant,
        'benign_malignant_confidence': float(benign_malignant_confidence)
    })

if __name__ == '__main__':
    app.run()
