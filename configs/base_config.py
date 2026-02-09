"""Base configuration for BBB prediction project."""
from pathlib import Path
import os


class BaseConfig:
    """Base settings for the BBB predictor project."""
    
    # Data paths
    DATA_DIR: Path = Path('data')
    TRAIN_DATA: Path = DATA_DIR / 'b3db_tanimoto_train.csv'
    TEST_DATA: Path = DATA_DIR / 'b3db_tanimoto_test.csv'
    VAL_DATA: Path = DATA_DIR / 'b3db_tanimoto_val.csv'
    
    # Target configuration
    TARGET_COLUMN: str = 'target'
    NUM_CLASSES: int = 2  # Binary classification for BBB permeability
    
    # Project settings
    PROJECT_NAME: str = "bbb-graph-predictor"
    EXPERIMENTS_DIR: Path = Path('experiments')
    
    @staticmethod
    def get_model_path(experiment_name: str) -> Path:
        """Get path to saved model file."""
        return BaseConfig.get_experiment_dir(experiment_name) / "model.pth"
    
    @staticmethod
    def get_experiment_dir(experiment_name: str) -> Path:
        """Get experiment directory, creating it if necessary."""
        exp_dir = BaseConfig.EXPERIMENTS_DIR / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir
    
    @staticmethod
    def get_results_path(experiment_name: str) -> Path:
        """Get path to results JSON file."""
        return BaseConfig.get_experiment_dir(experiment_name) / "results.json"
    
    @staticmethod
    def get_config_path(experiment_name: str) -> Path:
        """Get path to config JSON file."""
        return BaseConfig.get_experiment_dir(experiment_name) / "config.json"
