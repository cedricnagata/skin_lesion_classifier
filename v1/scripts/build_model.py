import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

def build_diag_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    diag_base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    diag_base_model.trainable = False

    diag_input = tf.keras.Input(shape=input_shape)
    diag_x = diag_base_model(diag_input, training=False)
    diag_x = layers.GlobalAveragePooling2D()(diag_x)

    diag_output = layers.Dense(3, activation='softmax', name='diagnosis')(diag_x)

    diag_model = models.Model(diag_input, diag_output)
    diag_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return diag_model

def build_bm_model(input_shape=(224, 224, 3), learning_rate=1e-3):
    bm_base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    bm_base_model.trainable = False

    bm_input = tf.keras.Input(shape=input_shape)
    bm_x = bm_base_model(bm_input, training=False)
    bm_x = layers.GlobalAveragePooling2D()(bm_x)

    bm_output = layers.Dense(1, activation='sigmoid', name='benign_malignant')(bm_x)

    bm_model = models.Model(bm_input, bm_output)
    bm_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return bm_model
