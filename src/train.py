import os
import yaml
import tensorflow as tf
from data_loader import get_data_generators
from model_builder import build_model
import numpy as np
import sys
import mlflow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

def calculate_class_weights(train_ds):
    """
    Calculates class weights for an imbalanced dataset.
    """
    print("Calculating class weights...")
    label_counts = {0: 0, 1: 0}
    
    for _, labels in train_ds:
        for label in labels.numpy():
            label_int = int(label[0])
            if label_int in label_counts:
                label_counts[label_int] += 1

    total_samples = sum(label_counts.values())
    if total_samples == 0:
        print("Warning: No samples found in train_ds. Returning default weights.")
        return {0: 1.0, 1: 1.0}

    normal_count = label_counts[0]
    pneumonia_count = label_counts[1]

    print(f"Total training samples: {total_samples}")
    print(f"NORMAL samples: {normal_count}")
    print(f"PNEUMONIA samples: {pneumonia_count}")

    num_classes = 2
    weight_for_0_normal = (total_samples / (num_classes * normal_count)) if normal_count > 0 else 1.0
    weight_for_1_pneumonia = (total_samples / (num_classes * pneumonia_count)) if pneumonia_count > 0 else 1.0

    class_weight_dict = {0: weight_for_0_normal, 1: weight_for_1_pneumonia}

    print(f"Weight for Class 0 (NORMAL): {weight_for_0_normal:.2f}")
    print(f"Weight for Class 1 (PNEUMONIA): {weight_for_1_pneumonia:.2f}")
    
    return class_weight_dict

def train_model():
    """
    Main training function, now with MLFlow logging.
    """
    try:
        with open(PARAMS_PATH, 'r') as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading {PARAMS_PATH}: {e}")
        sys.exit(1)

    data_params = params['data']
    model_params = params['model']
    output_params = params['output']
    mlflow_params = params['mlflow']

    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    mlflow.set_experiment(mlflow_params['experiment_name'])

    with mlflow.start_run():
        print("Starting MLFlow run...")
        
        print("Logging parameters...")
        mlflow.log_params(data_params)
        mlflow.log_params(model_params)

        print("\n--- Loading Data ---")
        train_ds, val_ds, test_ds, classes = get_data_generators(
            train_dir=data_params['train_dir'],
            test_dir=data_params['test_dir'],
            img_size=model_params['img_size'],
            batch_size=model_params['batch_size'],
            val_split=data_params['val_split'],
            seed=data_params['seed']
        )

        print("\n--- Building Model ---")
        model = build_model(img_size=model_params['img_size'])

        print("\n--- Handling Class Imbalance ---")
        class_weight_dict = calculate_class_weights(train_ds)
        
        mlflow.log_param("weight_for_normal", class_weight_dict[0])
        mlflow.log_param("weight_for_pneumonia", class_weight_dict[1])

        print("\n--- Starting Model Training ---")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=model_params['epochs'],
            class_weight=class_weight_dict
        )
        print("Training finished.")
        
        final_val_accuracy = history.history['val_accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        mlflow.log_metric("final_val_accuracy", final_val_accuracy)
        mlflow.log_metric("final_val_loss", final_val_loss)

        print("\n--- Saving Model ---")
        model_dir = os.path.dirname(os.path.join(PROJECT_ROOT, output_params['model_file']))
        os.makedirs(model_dir, exist_ok=True)
        
        model_path_abs = os.path.join(PROJECT_ROOT, output_params['model_file'])
        model.save(model_path_abs)
        print(f"Model saved successfully to {model_path_abs}")
        
        mlflow.log_artifact(model_path_abs, "model")
        
        print("MLFlow run finished.")

if __name__ == "__main__":
    train_model()