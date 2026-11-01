from pathlib import Path
from typing import List
import os

# CRITICAL: This will work both locally and on cluster
# On cluster, set: export B3PRED_DATA=/mnt/data
# Locally, you can set it to your data folder or leave as default
DATA_ROOT = Path(os.environ.get("B3PRED_DATA", "data"))

# For experiments folder, check if we're running in cluster (/mnt/code) or locally
# On cluster, results go to /mnt/results, locally to ./experiments
# Override with B3PRED_EXPERIMENTS_ROOT to redirect a run's outputs (e.g. seed sweeps)
if os.environ.get("B3PRED_EXPERIMENTS_ROOT"):
    EXPERIMENTS_ROOT = Path(os.environ["B3PRED_EXPERIMENTS_ROOT"])
elif os.path.exists("/mnt/code"):
    # Running on cluster
    EXPERIMENTS_ROOT = Path("/mnt/results/experiments")
else:
    # Running locally
    EXPERIMENTS_ROOT = Path("experiments")


class BaseSettings():
    # B3DB data paths
    TRAIN_DATA = DATA_ROOT / "b3db_tanimoto_train.csv"
    TEST_DATA = DATA_ROOT / "b3db_tanimoto_test.csv"
    VAL_DATA = DATA_ROOT / "b3db_tanimoto_val.csv"

    # MoleculeNet data paths
    MOLNET_TRAIN_DATA = DATA_ROOT / "moleculenet_tanimoto_train.csv"
    MOLNET_TEST_DATA = DATA_ROOT / "moleculenet_tanimoto_test.csv"
    MOLNET_VAL_DATA = DATA_ROOT / "moleculenet_tanimoto_val.csv"

    # SData (cross-dataset test)
    SDATA_TEST_DATA = DATA_ROOT / "sdata.csv"
    SDATA_TARGET_COL: str = "BBB"

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
