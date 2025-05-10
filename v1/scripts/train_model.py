import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, train_gen, val_gen, class_weight, epochs=100, models_path=None, model_name=None):
    path = os.path.join(models_path, model_name)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(path, monitor='val_loss', save_best_only=True, mode='min')
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=[checkpoint, early_stopping, lr_scheduler]
    )
