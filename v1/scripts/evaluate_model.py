import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, model_name, test_ds, results_path=None, results_file=None):
    # Evaluate the model
    evaluation = model.evaluate(test_ds)
    metrics = dict(zip(model.metrics_names, evaluation))
    
    # Get true labels and predictions
    y_true = []
    y_pred = []

    for batch in test_ds:
        images, labels = batch
        if model_name == 'diagnosis':
            y_true.extend(np.argmax(labels, axis=1))
            predictions = model.predict_on_batch(images)
            y_pred.extend(np.argmax(predictions, axis=1))
        elif model_name == 'benign_malignant':
            y_true.extend(labels)
            predictions = model.predict_on_batch(images)
            y_pred.extend((predictions > 0.5).astype(int))

    # Compute classification report
    if model_name == 'diagnosis':
        report = classification_report(y_true, y_pred, target_names=['nevus', 'melanoma', 'other'], output_dict=True)
    elif model_name == 'benign_malignant':
        report = classification_report(y_true, y_pred, target_names=['benign', 'malignant'], output_dict=True)
    
    metrics.update(report)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    os.makedirs(results_path, exist_ok=True)
    if results_file:
        with open(os.path.join(results_path, results_file), 'w') as f:
            for metric, value in metrics.items():
                f.write(f'{metric}: {value}\n')
            
            f.write(f'\n{model_name.capitalize()} Confusion Matrix:\n')
            f.write(np.array2string(cm))
