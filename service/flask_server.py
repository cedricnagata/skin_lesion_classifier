from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

TFLITE = False

if TFLITE:
    # Load the TFLite models
    interpreter_3_features = tf.lite.Interpreter(model_path="model_2.tflite")
    interpreter_34_features = tf.lite.Interpreter(model_path="model_34.tflite")
    interpreter_3_features.allocate_tensors()
    interpreter_34_features.allocate_tensors()
else:
    # Load the Keras models
    model_3_features = tf.keras.models.load_model('model_2.keras')
    model_34_features = tf.keras.models.load_model('model_34.keras')

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
    if diagnosis:
        preprocessor_34_features = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['age_approx']),
                ('cat', OneHotEncoder(
                    categories=[['male', 'female'], list(DIAGNOSIS_MAPPING.keys())], 
                    sparse_output=False), ['sex', 'diagnosis']),
            ]
        )
        preprocessor_34_features.fit(pd.DataFrame({
            'age_approx': [0],
            'sex': ['male'],
            'diagnosis': ['AIMP']
        }))
        data = pd.DataFrame({'age_approx': [float(age)], 'sex': [sex.lower()], 'diagnosis': [diagnosis]})
        features_preprocessed = preprocessor_34_features.transform(data)
    else:
        preprocessor_3_features = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), ['age_approx']),
                ('cat', OneHotEncoder(
                    categories=[['male', 'female']], 
                    sparse_output=False), ['sex']),
            ]
        )
        preprocessor_3_features.fit(pd.DataFrame({
            'age_approx': [0],
            'sex': ['male']
        }))
        data = pd.DataFrame({'age_approx': [float(age)], 'sex': [sex.lower()]})
        features_preprocessed = preprocessor_3_features.transform(data)
    
    features_preprocessed = features_preprocessed.flatten().reshape(1, -1).astype('float32')
    return features_preprocessed


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
            model = interpreter_34_features if TFLITE else model_34_features
            tabular_data = preprocess_tabular_data(age, sex, diagnosis)
        else:
            model = interpreter_3_features if TFLITE else model_3_features
            tabular_data = preprocess_tabular_data(age, sex)

        if TFLITE:
            # Prepare input details
            input_details = model.get_input_details()

            # Set input tensor
            model.set_tensor(input_details[0]['index'], image.astype(np.float32))
            model.set_tensor(input_details[1]['index'], tabular_data.astype(np.float32))
            model.invoke()

            # Get predictions
            output_details = model.get_output_details()
            predictions = model.get_tensor(output_details[0]['index'])
        else:
            predictions = model.predict([image, tabular_data])

        prediction_probability = predictions[0][0]
        predicted_class = 1 if prediction_probability > 0.5 else 0

        # Calculate the confidence as the absolute distance from 0.5
        confidence = abs(prediction_probability - 0.5) * 2

        return jsonify({
            'prediction': PREDICTION_MAPPING[predicted_class],  # Map prediction to class name
            'confidence': float(confidence)
        })
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
