import h5py
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight

def create_datasets(train_file, val_file, test_file, batch_size):
    train_images, train_diagnoses, train_benign_malignants = load_h5_data(train_file)
    val_images, val_diagnoses, val_benign_malignants = load_h5_data(val_file)
    test_images, test_diagnoses, test_benign_malignants = load_h5_data(test_file)

    """
    # Recalculate class weights after augmentation
    diagnosis_weights = calculate_class_weights(np.argmax(train_diagnoses, axis=1), 'diagnosis')
    benign_malignant_weights = calculate_class_weights(train_benign_malignants, 'benign_malignant')
    """

    # Identify the minority classes (assuming 0: nevus, 1: melanoma, 2: other)
    minority_classes_dn = [1, 2]

    train_images_aug, train_dn_aug, train_bm_aug = augment_minority_classes(
        train_images, train_diagnoses, train_benign_malignants, minority_classes_dn
    )

    # Create datasets
    #train_dn = augment_dataset(train_images, train_diagnoses, batch_size)
    train_dn = tf.data.Dataset.from_tensor_slices((train_images_aug, train_dn_aug)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dn = tf.data.Dataset.from_tensor_slices((val_images, val_diagnoses)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_dn = tf.data.Dataset.from_tensor_slices((test_images, test_diagnoses)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    #train_bm = augment_dataset(train_images, train_benign_malignants, batch_size)
    train_bm = tf.data.Dataset.from_tensor_slices((train_images_aug, train_bm_aug)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_bm = tf.data.Dataset.from_tensor_slices((val_images, val_benign_malignants)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_bm = tf.data.Dataset.from_tensor_slices((test_images, test_benign_malignants)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Recalculate class weights after augmentation
    diagnosis_weights = calculate_class_weights(np.argmax(train_dn_aug, axis=1), 'diagnosis')
    benign_malignant_weights = calculate_class_weights(train_bm_aug, 'benign_malignant')

    return train_dn, val_dn, test_dn, train_bm, val_bm, test_bm, diagnosis_weights, benign_malignant_weights

def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image

def augment_dataset(images, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(len(images)).batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def augment_minority_classes(images, diagnosis_labels, bm_labels, minority_classes, aug_factor=5):
    augmented_images = []
    augmented_diagnosis_labels = []
    augmented_bm_labels = []
    for image, diag_label, bm_label in zip(images, diagnosis_labels, bm_labels):
        augmented_images.append(image)
        augmented_diagnosis_labels.append(diag_label)
        augmented_bm_labels.append(bm_label)
        if np.argmax(diag_label) in minority_classes:
            for _ in range(aug_factor):
                augmented_images.append(augment_image(image))
                augmented_diagnosis_labels.append(diag_label)
                augmented_bm_labels.append(bm_label)
    return np.array(augmented_images), np.array(augmented_diagnosis_labels), np.array(augmented_bm_labels)

def calculate_class_weights(labels, label_type):
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = dict(enumerate(class_weights))
    return class_weights_dict

def load_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        diagnoses = f['diagnoses'][:]
        benign_malignants = f['benign_malignants'][:]
    return images, diagnoses, benign_malignants
