import os
import json
import sys
import yaml
import tensorflow as tf
import mlflow

from src.data.loader import get_data_generators
from src.training.model import build_model


# Path resolution (DO NOT CHANGE)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

PARAMS_PATH = os.path.join(PROJECT_ROOT, "configs", "params.yaml")
DATA_ROOT = os.path.join(PROJECT_ROOT, "artifacts", "data_processed")



# Evaluation logic


def evaluate_model():
    """Evaluate trained model on test set and log metrics."""

    try:
        with open(PARAMS_PATH, "r") as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load params.yaml: {e}")
        sys.exit(1)

    data_cfg = params["data"]
    model_cfg = params["model"]
    output_cfg = params["output"]
    mlflow_cfg = params["mlflow"]

    # Resolve paths
    test_dir = os.path.join(DATA_ROOT, data_cfg["test_dir"])
    model_path = os.path.join(PROJECT_ROOT, output_cfg["model_file"])

    print(f"[INFO] Loading test data from: {test_dir}")

    _, _, test_ds, class_names = get_data_generators(
        train_dir=None,                # not used in evaluation
        test_dir=test_dir,
        img_size=model_cfg["img_size"],
        batch_size=model_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        seed=data_cfg["seed"],
    )

    print(f"[INFO] Classes: {class_names}")

    print(f"[INFO] Loading model from: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    run_id_path = os.path.join(PROJECT_ROOT, "run_id.json")
    run_id = None

    if os.path.exists(run_id_path):
        with open(run_id_path, "r") as f:
            run_id = json.load(f).get("run_id")
        print(f"[INFO] Resuming MLflow run: {run_id}")
    else:
        print("[INFO] No training run_id found. Starting new evaluation run.")

    with mlflow.start_run(run_id=run_id):
        if run_id is None:
            mlflow.set_tag("run_type", "evaluation_only")

        print("[INFO] Evaluating model on test set...")
        results = model.evaluate(test_ds, verbose=1)

        metrics = {
            "test_loss": float(results[0]),
            "test_accuracy": float(results[1]),
            "test_precision": float(results[2]),
            "test_recall": float(results[3]),
        }

        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        metrics_dir = os.path.join(PROJECT_ROOT, "artifacts", "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"[INFO] Metrics saved to: {metrics_path}")

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(metrics_path, artifact_path="metrics")

        print("[INFO] Evaluation completed successfully.")


if __name__ == "__main__":
    evaluate_model()
