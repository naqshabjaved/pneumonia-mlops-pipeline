import shutil
from pathlib import Path

RAW_DATA_DIR = Path("data")
PROCESSED_DATA_DIR = Path("artifacts/data_processed")

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if RAW_DATA_DIR.exists():
    shutil.copytree(RAW_DATA_DIR, PROCESSED_DATA_DIR, dirs_exist_ok=True)
