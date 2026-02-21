from pathlib import Path
from typing import List
import os

# CRITICAL: This will work both locally and on cluster
# On cluster, set: export B3PRED_DATA=/mnt/data
# Locally, you can set it to your data folder or leave as default
DATA_ROOT = Path(os.environ.get("B3PRED_DATA", "data"))

# For experiments folder, check if we're running in cluster (/mnt/code) or locally
# On cluster, results go to /mnt/results, locally to ./experiments
if os.path.exists("/mnt/code"):
    # Running on cluster
    EXPERIMENTS_ROOT = Path("/mnt/results/experiments")
else:
    # Running locally
    EXPERIMENTS_ROOT = Path("experiments")


class BaseSettings():
    # Data paths - will use B3PRED_DATA environment variable
    TRAIN_DATA = DATA_ROOT / "b3db_tanimoto_train.csv"
    TEST_DATA = DATA_ROOT / "b3db_tanimoto_test.csv"
    VAL_DATA = DATA_ROOT / "b3db_tanimoto_val.csv"
    
    TARGET_LABEL: str = 'target'
    TEST_LABEL: str = 'target'
    PROJECT_NAME: str = "b3db-graph-models"

    # Optimization strategy settings
    OPT_SUBSET_SIZE = 0.1  # Use 10% of data for optimization
    OPT_EPOCHS = 10  # Train for 10 epochs during optimization
    TOP_K_MODELS = 10  # Number of top models to retrain
    FULL_EPOCHS = 100  # Full training epochs
    
    # Experiments folder - automatically adapts to cluster or local
    EXPERIMENTS_FOLDER: Path = EXPERIMENTS_ROOT

    @staticmethod
    def get_model_path(model_name: str) -> Path:
        model_path = BaseSettings.get_model_folder(model_name) / "model.pth"
        return model_path

    @staticmethod
    def get_model_folder(model_name: str) -> Path:
        model_path = BaseSettings.EXPERIMENTS_FOLDER / model_name

        # create folder if it doesn't exist
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)

        return model_path
