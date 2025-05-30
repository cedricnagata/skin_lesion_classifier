import modal

app = modal.App("skin-lesion-classifier")
image = (
    modal.Image.debian_slim()
    .pip_install("flask", "tensorflow", "keras", "numpy", "pillow")
)
volume = modal.Volume.from_name("models")
VOL_DIR = "/models"

@app.function(image=image, volumes={"/models": volume})
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def flask_app():
    from flask import Flask, request, jsonify
    import tensorflow as tf
    import keras
    from keras import layers, applications, Model, optimizers
    import numpy as np
    from PIL import Image
    import io

    app = Flask(__name__)

    IMG_SIZE = 384

    # Load the model
    try:
        print(f"Loading model...")

        inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        model = applications.EfficientNetV2L(include_top=False, input_tensor=inputs)
        model.trainable = False
        # Rebuild top
        x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
        x = layers.BatchNormalization()(x)
        top_dropout_rate = 0.2
        x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="pred")(x)
        # Compile
        model = Model(inputs, outputs, name="EfficientNet")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-2),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        model.load_weights(f"{VOL_DIR}/slc_85.weights.h5")
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

    def preprocess_image(image_file):
        img_bytes = io.BytesIO(image_file)
        img = tf.keras.utils.load_img(
            img_bytes,
            target_size=(IMG_SIZE, IMG_SIZE),
            interpolation="bicubic"
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    @app.post("/predict")
    def predict():
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
            
        image_file = request.files['image'].read()
        try:
            # Preprocess the image
            processed_image = preprocess_image(image_file)
            prediction = model.predict(processed_image)[0][0]

            # Convert prediction to class label
            result = {
                "prediction": float(prediction),
                "class": "malignant" if prediction > 0.5 else "benign",
                "confidence": float(prediction if prediction > 0.5 else 1 - prediction),
            }

            return jsonify(result)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return jsonify({"error": str(e)}), 500

    return app