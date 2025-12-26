# Pneumonia Detection – MLOps-Oriented ML Pipeline

## Problem Statement

This project addresses **binary classification of chest X-ray images** into **NORMAL** and **PNEUMONIA**, with a strong emphasis on **high recall** to minimize false negatives in a medical screening context.

Beyond model accuracy, the primary goal of this repository is to demonstrate **reproducible machine learning workflows** and **ML lifecycle discipline** using modern MLOps tooling.

---

## Current Scope & Intent

This repository implements a **reproducible training and evaluation pipeline** for a computer vision model using **DVC-based pipeline orchestration**.

> **Important:**  
> This project is intentionally scoped as a **training–evaluation MLOps pipeline**.  
> It does **not yet** implement full production serving, monitoring, or model governance. Those are planned extensions.

This scoping choice is deliberate and documented to ensure architectural clarity and reproducibility.

---

## Architecture Overview

**Current lifecycle implemented:**
```text
Versioned Data → Training → Evaluation → Tracked Artifacts
```

The pipeline is defined as a **Directed Acyclic Graph (DAG)** using `dvc.yaml`, enabling deterministic re-runs and experiment comparison.

---

## Pipeline Stages

### 1. Training
- Transfer learning using **MobileNetV2**
- Parameterized via `params.yaml`
- Model artifact saved as a versioned output

### 2. Evaluation
- Evaluation on held-out test data
- Metrics logged for model comparison
- Recall prioritized due to medical domain requirements

---

## Model & Results (Latest Run)

| Metric | Result | Context |
|------|--------|--------|
| **Test Accuracy** | **89.42%** | Overall classification performance |
| **Test Recall** | **93.33%** | Critical for minimizing missed pneumonia cases |
| **Test Precision** | **91.43%** | Controls false positives |

---

## Tooling Stack

| Component | Tool | Role in System |
|---------|------|---------------|
| Pipeline Orchestration | DVC | Defines reproducible ML stages |
| Configuration | YAML | Parameterized experimentation |
| Model Architecture | MobileNetV2 | Transfer learning backbone |
| Artifact Tracking | DVC outputs | Models & metrics versioning |
| Demo Inference | Streamlit | Local interactive prediction demo |

---

## What Is Implemented (Phase 0)

- Deterministic ML pipeline via DVC  
- Parameterized training and evaluation  
- Versioned model artifacts  
- Versioned evaluation metrics  
- Local inference demo for qualitative validation  

---

## What Is Deliberately Out of Scope (Phase 0)

The following are **not yet implemented** and are explicitly deferred:

- Model registry & promotion workflow  
- Production inference API (FastAPI / REST)  
- Monitoring, drift detection, and alerting  
- Automated cloud deployment  
- CI/CD-driven retraining and release gates  

This separation ensures the current system remains **focused, testable, and reproducible**.

---

## Roadmap to Production

Planned future extensions include:

### Phase 1
- Explicit data preparation & validation stage  
- Artifact lifecycle separation (staging vs production)  
- Structured experiment tracking  

### Phase 2
- REST-based inference service  
- Containerized deployment  
- Monitoring and model performance tracking  
- Automated CI/CD enforcement  

---

## How to Run Locally
### 1. Clone the Repository
```bash
git clone <repo-url>
cd pneumonia-mlops-pipeline
```
### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate          # Windows
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Acquire Dataset
Download the Chest X-ray Pneumonia dataset and place it in the following structure:
```text
data/
└── chest_xray/
    ├── train/
    └── test/
```
The dataset is intentionally excluded from version control and must be provided externally.
### 5. Reproduce the ML Pipeline
Run the complete training and evaluation workflow using DVC:
```bash
dvc repro
```
This will:

- Execute training and evaluation stages
- Generate a versioned model artifact
- Produce evaluation metrics

### 6. Run Local Inference Demo
Launch the interactive inference interface:
```bash 
streamlit run app.py
```
Upload a chest X-ray image to obtain a pneumonia prediction.
### 7. Optional: Inspect Artifacts

After execution, the following artifacts will be available locally:

- Trained model: models/
- Evaluation metrics: metrics.json

These artifacts are reproducible and traceable through the DVC pipeline.

## Author

**Naqshab Javed**
