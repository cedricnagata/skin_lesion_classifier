from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gc

# Define the labels
diagnosis_labels = ['nevus', 'melanoma', 'other']
benign_malignant_labels = ['benign', 'malignant']

# Global variables for models
diagnosis_model = None
benign_malignant_model = None

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

def create_app():
    app = Flask(__name__)
    
    # Load models when creating the app
    global diagnosis_model, benign_malignant_model
    print("Starting model initialization...")
    print("Loading diagnosis model...")
    diagnosis_model = load_model_with_optimization('diagnosis_model.h5')
    print("Diagnosis model loaded successfully!")
    print("Loading benign/malignant model...")
    benign_malignant_model = load_model_with_optimization('benign_malignant_model.h5')
    print("Benign/malignant model loaded successfully!")
    print("All models initialized successfully!")
    
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

        # Make predictions using the loaded models
        diagnosis_predictions = diagnosis_model.predict(img_array)
        benign_malignant_predictions = benign_malignant_model.predict(img_array)

        # Process diagnosis predictions
        diagnosis_pred = np.argmax(diagnosis_predictions, axis=1)[0]
        diagnosis_confidence = diagnosis_predictions[0][diagnosis_pred]

        # Process benign/malignant predictions
        benign_malignant_pred = int(benign_malignant_predictions[0] > 0.5)
        benign_malignant_confidence = benign_malignant_predictions[0][0] if benign_malignant_pred else 1 - benign_malignant_predictions[0][0]

        # Map predictions to labels
        diagnosis = diagnosis_labels[diagnosis_pred]
        benign_malignant = benign_malignant_labels[benign_malignant_pred]

        # Remove the temporary file
        os.remove(temp_file_path)

        # Return the predictions
        return jsonify({
            'diagnosis': diagnosis,
            'diagnosis_confidence': float(diagnosis_confidence),
            'benign_malignant': benign_malignant,
            'benign_malignant_confidence': float(benign_malignant_confidence)
        })
    
    return app

app = create_app()

if __name__ == '__main__':
    app.run()
