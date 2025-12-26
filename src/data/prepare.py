from pathlib import Path
import shutil

# Resolve paths RELATIVE TO REPO ROOT
REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = REPO_ROOT / "artifacts" / "data_processed"

# Ensure output directory exists
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Copy raw data if it exists
if RAW_DATA_DIR.exists():
    shutil.copytree(
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        dirs_exist_ok=True
    )
else:
    raise FileNotFoundError("Raw data directory not found: data/")
