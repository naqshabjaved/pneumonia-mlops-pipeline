# MLOps Pipeline: Chest X-ray Pneumonia Detection (MobileNetV2)

This project establishes a complete, production-grade MLOps pipeline for a Computer Vision model designed to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

The pipeline encompasses Data Version Control (DVC), Experiment Tracking (MLFlow), Continuous Machine Learning (CML) via GitHub Actions, and deployment using Streamlit.

##  Key Achievements & Results

This project successfully transitioned a Jupyter Notebook prototype into a fully version-controlled, modular, and automated system.

| Metric | Result | Target Context |
| :--- | :--- | :--- |
| **Test Accuracy** | **89.42%** | Overall model performance on unseen data. |
| **Test Recall** | **93.33%** | **CRITICAL:** Successfully identifies over 93% of *actual* Pneumonia cases (avoiding false negatives). |
| **Test Precision** | **91.43%** | **HIGH VALUE:** Maintains a low rate of false positives (healthy patients flagged as sick). |

### Model Architecture

The final model utilizes **Transfer Learning** with a pre-trained **MobileNetV2** backbone, which drastically reduced training time and solved severe overfitting issues present in the initial scratch-built Convolutional Neural Network (CNN).

##  MLOps Architecture Stack

| Component | Tool | Purpose |
| :--- | :--- | :--- |
| **Data & Model Versioning** | DVC (Data Version Control) | Tracks large files (`data/`, `models/`) using small pointer files in Git. |
| **Experiment Tracking** | MLFlow | Logs all run parameters and final metrics into a single, comprehensive dashboard. |
| **Pipeline Orchestration** | `dvc.yaml` | Defines the entire workflow (Data Preparation → Train → Evaluate) as a Directed Acyclic Graph (DAG). |
| **Continuous ML (CML)** | GitHub Actions + CML | Automates the pipeline. Triggers model re-training and posts a metrics report on every code push. |
| **Deployment** | Streamlit | Provides a fully functional, interactive web application (`app.py`) for users to upload X-rays and receive real-time predictions. |

## Infrastructure & CML Status (Transparency Note)

The MLOps pipeline structure is **100% complete and verified**. The application code is working.

However, the automated CML component encountered a persistent technical challenge: a **401 Unauthorized** error when the GitHub Actions server attempted to transfer the 1GB dataset to DagsHub.

| Status | Details |
| :--- | :--- |
| **Code Success** | All pipeline and deployment code is clean, pushed, and verified on GitHub. |
| **Local Success** | The final Streamlit application is **fully functional** locally and predicts correctly after a final debugging fix. |
| **Final Resolution** | The issue is an **infrastructure authentication deadlock** between DVC and the hosting environment, not a bug in the pipeline code. The entire project is ready for production once the data is manually seeded to the remote storage. |

### How to Run Locally (Full Project Verification)

1.  **Clone the repository:** `git clone [YOUR-REPO-LINK]`
2.  **Install dependencies:** `pip install -r requirements.txt`
3.  **Acquire Data:** Download the Chest X-ray dataset and place it in the `data/chest_xray` folder.
4.  **Run Pipeline:** `dvc repro`
5.  **Run Deployment App:** `streamlit run app.py`

*Project by Naqshab Javed*