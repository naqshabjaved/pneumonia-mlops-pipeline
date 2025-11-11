import tensorflow as tf
import os
import yaml
import json
import sys
from data_loader import get_data_generators
import mlflow 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PARAMS_PATH = os.path.join(PROJECT_ROOT, "params.yaml")

def evaluate_model():
    """
    Loads the trained model, evaluates it, saves metrics to JSON,
    and logs metrics to MLFlow.
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
    
    model_path_abs = os.path.join(PROJECT_ROOT, output_params['model_file'])

    print("\n--- Loading Test Data ---")
    _, _, test_ds, classes = get_data_generators(
        train_dir=data_params['train_dir'],
        test_dir=data_params['test_dir'],
        img_size=model_params['img_size'],
        batch_size=model_params['batch_size'],
        val_split=data_params['val_split'],
        seed=data_params['seed']
    )

    print(f"\n--- Loading Model From {model_path_abs} ---")
    try:
        model = tf.keras.models.load_model(model_path_abs)
    except Exception as e:
        print(f"Error loading model from {model_path_abs}: {e}")
        print("Please ensure you have run 'python src/train.py' successfully first.")
        sys.exit(1)
    print("Model loaded successfully.")

    mlflow.set_tracking_uri(mlflow_params['tracking_uri'])
    mlflow.set_experiment(mlflow_params['experiment_name'])
 
    with mlflow.start_run(run_name="EvaluationRun", nested=True):
        print("\n--- Evaluating Model on Test Set ---")
        results = model.evaluate(test_ds)

        test_loss = results[0]
        test_accuracy = results[1]
        test_precision = results[2]
        test_recall = results[3]

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({(test_accuracy*100):.2f}%)")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
        
        metrics_path = os.path.join(PROJECT_ROOT, 'metrics.json')
        
        try:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"\nMetrics saved successfully to {metrics_path}")
            
            print("Logging metrics to MLFlow...")
            mlflow.log_metrics(metrics)
            
            mlflow.log_artifact(metrics_path, "metrics")
            print("MLFlow evaluation run finished.")
            
        except Exception as e:
            print(f"Error saving/logging metrics: {e}")

if __name__ == "__main__":
    evaluate_model()