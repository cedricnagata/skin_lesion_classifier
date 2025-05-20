from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = Flask(__name__)

# Get model path from environment variable
MODEL_PATH = "skin_lesion_classifier_85.keras"

# Load the model
try:
    print(f"Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    if img.size != (384, 384):
        print(f"Resizing image from {img.size} to (384, 384)")
        img = img.resize((384, 384), resample=Image.Resampling.BICUBIC)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():        
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Get the image file from the form data
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Preprocess the image
        processed_image = preprocess_image(image_bytes)
        prediction = model.predict(processed_image)[0][0]
        
        # Convert prediction to class label
        result = {
            'prediction': float(prediction),
            'class': 'malignant' if prediction > 0.5 else 'benign',
            'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000) 