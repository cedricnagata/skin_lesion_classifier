from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown

app = Flask(__name__)

MODEL_2_PATH = 'model_2.tflite'
MODEL_34_PATH = 'model_34.tflite'

"""
KERAS VERSION
# Load your trained model
# model = tf.keras.models.load_model(MODEL_PATH)
"""

# ****TFLITE VERSION****
# Load TFLite models
interpreter_without_diagnosis = tf.lite.Interpreter(model_path=MODEL_2_PATH)
interpreter_with_diagnosis = tf.lite.Interpreter(model_path=MODEL_34_PATH)

interpreter_without_diagnosis.allocate_tensors()
interpreter_with_diagnosis.allocate_tensors()
# ****TFLITE VERSION****

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

def preprocess_tabular_data(age, sex, diagnosis=None):
    age = np.array([float(age)], dtype=np.float32).reshape(1, 1)
    if sex == 'male':
        sex = np.array([[1, 0]], dtype=np.float32)
    elif sex == 'female':
        sex = np.array([[0, 1]], dtype=np.float32)
    else:
        raise ValueError('Sex must be "male" or "female"')

    if diagnosis and diagnosis in DIAGNOSIS_MAPPING:
        diagnosis_encoded = DIAGNOSIS_MAPPING[diagnosis]
        one_hot_diagnosis = np.zeros((1, len(DIAGNOSIS_MAPPING)), dtype=np.float32)
        one_hot_diagnosis[0, diagnosis_encoded] = 1.0
        features = np.concatenate((age, sex, one_hot_diagnosis), axis=1)
    else:
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
        diagnosis = request.form.get('diagnosis')

        if diagnosis and diagnosis in DIAGNOSIS_MAPPING:
            interpreter = interpreter_with_diagnosis
            tabular_data = preprocess_tabular_data(age, sex, diagnosis)
        else:
            interpreter = interpreter_without_diagnosis
            tabular_data = preprocess_tabular_data(age, sex)

        # Prepare input details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.set_tensor(input_details[1]['index'], tabular_data)

        # Run inference
        interpreter.invoke()

        # Get predictions
        predictions = interpreter.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(predictions, axis=1)[0]
        prediction_confidence = predictions[0][predicted_class]

        return jsonify({
            'prediction': PREDICTION_MAPPING[predicted_class],  # Map prediction to class name
            'confidence': float(prediction_confidence)
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
