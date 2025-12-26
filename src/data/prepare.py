from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = REPO_ROOT / "data"
PROCESSED_DATA_DIR = REPO_ROOT / "artifacts" / "data_processed"

def main():
    if not RAW_DATA_DIR.exists():
        print(f"[FATAL] Raw data directory not found: {RAW_DATA_DIR}")
        sys.exit(1)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Copy CONTENTS of data/, not the data/ directory itself
    for item in RAW_DATA_DIR.iterdir():
        dest = PROCESSED_DATA_DIR / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"[INFO] Data prepared at: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    main()
