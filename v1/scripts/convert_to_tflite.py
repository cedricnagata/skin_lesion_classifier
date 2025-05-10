import tensorflow as tf

def convert_to_tflite(model_h5_path, tflite_model_path):
    # Load the Keras model from the .h5 file
    model = tf.keras.models.load_model(model_h5_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

if __name__ == "__main__":
    # Convert the diagnosis model
    convert_to_tflite(
        'C:/Users/thece/Documents/Software Projects/DERM DX/data/models/diagnosis_model.h5', 
        'C:/Users/thece/Documents/Software Projects/DERM DX/data/models/diagnosis_model.tflite'
    )

    # Convert the benign/malignant model
    convert_to_tflite(
        'C:/Users/thece/Documents/Software Projects/DERM DX/data/models/benign_malignant_model.h5', 
        'C:/Users/thece/Documents/Software Projects/DERM DX/data/models/benign_malignant_model.tflite'
    )
