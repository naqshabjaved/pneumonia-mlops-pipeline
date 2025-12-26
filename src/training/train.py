from pathlib import Path
import yaml
import json
import sys
import tensorflow as tf
import mlflow

from src.data.loader import get_data_generators
from src.training.model import build_model



# Path Resolution (Repo-root safe, execution-context independent)


REPO_ROOT = Path(__file__).resolve().parents[2]

PARAMS_PATH = REPO_ROOT / "configs" / "params.yaml"
DATA_ROOT = REPO_ROOT / "artifacts" / "data_processed"
MODEL_DIR = REPO_ROOT / "artifacts" / "models"
METRICS_DIR = REPO_ROOT / "artifacts" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# Utility: Class Weights


def calculate_class_weights(train_ds):
    """
    Compute class weights for imbalanced binary classification.
    """
    label_counts = {0: 0, 1: 0}

    for _, labels in train_ds:
        for label in labels.numpy():
            label_counts[int(label[0])] += 1

    total = sum(label_counts.values())
    if total == 0:
        return {0: 1.0, 1: 1.0}

    return {
        0: total / (2 * label_counts[0]) if label_counts[0] > 0 else 1.0,
        1: total / (2 * label_counts[1]) if label_counts[1] > 0 else 1.0,
    }



# Training Entry Point


def train():
    
    # Load configuration
    
    try:
        with open(PARAMS_PATH, "r") as f:
            params = yaml.safe_load(f)
    except Exception as e:
        print(f"[FATAL] Failed to load params.yaml: {e}")
        sys.exit(1)

    data_cfg = params["data"]
    model_cfg = params["model"]
    output_cfg = params["output"]
    mlflow_cfg = params.get("mlflow", {})

    
    # MLflow setup (tracking only)
    
    if mlflow_cfg:
        mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run() if mlflow_cfg else nullcontext() as run:
        if run:
            mlflow.log_params(model_cfg)
            mlflow.log_params(data_cfg)

        
        train_ds, val_ds, test_ds, classes = get_data_generators(
            train_dir=DATA_ROOT
