from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

# URL of the shared Google Drive files
MODEL_2_URL = 'https://drive.google.com/uc?id=14vdMeLs6QDNHrYtkrwU3NgNjdm-oVQMJ'
MODEL_2_PATH = 'model_2.tflite'
MODEL_34_URL = 'https://drive.google.com/uc?id=1WyaO0boMXI-2IPU0bLsdRlUelkSBlDH6'
MODEL_34_PATH = 'model_34.tflite'

def download_model():
    if not os.path.exists(MODEL_2_PATH):
        gdown.download(MODEL_2_URL, MODEL_2_PATH, quiet=False)

# Download the model
download_model()

"""
KERAS VERSION
# Load your trained model
# model = tf.keras.models.load_model(MODEL_PATH)
"""

# ****TFLITE VERSION****
# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='model_2.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# ****TFLITE VERSION****

"""
DIAGNOSIS_MAPPING = {
    'AIMP': 0,
    'acrochordon': 1,
    'actinic keratosis': 2,
    'angiofibroma or fibrous papule': 3,
    'angiokeratoma': 4,
    'angioma': 5,
    'atypical melanocytic proliferation': 6,
    'basal cell carcinoma': 7,
    'cafe-au-lait macule': 8,
    'clear cell acanthoma': 9,
    'dermatofibroma': 10,
    'lentigo NOS': 11,
    'lentigo simplex': 12,
    'lichenoid keratosis': 13,
    'melanoma': 14,
    'melanoma metastasis': 15,
    'mucosal melanosis': 16,
    'neurofibroma': 17,
    'nevus': 18,
    'nevus spilus': 19,
    'other': 20,
    'pigmented benign keratosis': 21,
    'pyogenic granuloma': 22,
    'scar': 23,
    'sebaceous adenoma': 24,
    'sebaceous hyperplasia': 25,
    'seborrheic keratosis': 26,
    'solar lentigo': 27,
    'squamous cell carcinoma': 28,
    'vascular lesion': 29,
    'verruca': 30,
}

# Define a default diagnosis
DEFAULT_DIAGNOSIS = 'nevus'
"""

# Define the mapping for predictions
PREDICTION_MAPPING = {
    0: 'benign',
    1: 'malignant'
}

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def preprocess_tabular_data(age, sex):
    age = np.array([float(age)], dtype=np.float32).reshape(1, -1)

    if sex == 'male':
        sex = np.array([[1, 0]], dtype=np.float32)
    elif sex == 'female':
        sex = np.array([[0, 1]], dtype=np.float32)
    else:
        raise ValueError('Sex must be "male" or "female"')
    
    features = np.concatenate((age, sex), axis=1)
    return features

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    if not all(k in request.form for k in ('age', 'sex')):
        return jsonify({'error': 'Missing tabular data in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Preprocess image
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        # Preprocess tabular data
        age = request.form['age']
        sex = request.form['sex']
        tabular_data = preprocess_tabular_data(age, sex)

        """
        KERAS VERSION
        # Perform prediction
        predictions = model.predict([image, tabular_data])
        """

        # ****TFLITE VERSION****
        # Set the tensor for the image
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.set_tensor(input_details[1]['index'], tabular_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        # ****TFLITE VERSION****

        predicted_class = np.argmax(predictions, axis=1)[0]
        prediction_confidence = predictions[0][predicted_class]

        return jsonify({
            'prediction': PREDICTION_MAPPING[predicted_class],  # Map prediction to class name
            'confidence': float(prediction_confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
