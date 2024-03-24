from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, concatenate, Dropout
from tensorflow.keras.optimizers import Adam

def create_image_branch(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    return inputs, x

def create_tabular_branch(input_shape=(34,)):  # Adjust the number of tabular features if necessary
    inputs = Input(shape=input_shape)
    x = Dense(64, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    return inputs, x

def create_multi_input_model(image_input_shape=(224, 224, 3), tabular_input_shape=(34,)):
    image_inputs, image_branch = create_image_branch(image_input_shape)
    tabular_inputs, tabular_branch = create_tabular_branch(tabular_input_shape)
    concatenated = concatenate([image_branch, tabular_branch])
    
    x = Dense(64, activation='relu')(concatenated)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=[image_inputs, tabular_inputs], outputs=outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model
