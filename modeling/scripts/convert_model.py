import tensorflow as tf

# Load your trained model
model = tf.keras.models.load_model('model_34.keras')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable optimizations like quantization

try:
    tflite_model = converter.convert()
    with open('model_34.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted successfully.")
except Exception as e:
    print(f"Error converting model: {e}")
