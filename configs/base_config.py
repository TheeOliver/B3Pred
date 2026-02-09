from pathlib import Path
from typing import List
import os

class BaseSettings():
    # Update these paths to match your system
    TRAIN_DATA: Path = Path('data/b3db_tanimoto_train.csv')
    TEST_DATA: Path = Path('data/b3db_tanimoto_test.csv')
    VAL_DATA: Path = Path('data/b3db_tanimoto_val.csv')
    TARGET_LABEL: str = 'target'
    TEST_LABEL: str = 'target'
    PROJECT_NAME: str = "b3db-graph-models"
    EXPERIMENTS_FOLDER: Path = Path('experiments')

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
