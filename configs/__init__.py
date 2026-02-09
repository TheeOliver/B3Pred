"""Configuration package for BBB predictor."""
from configs.base_config import BaseConfig
from configs.model_configs import (
    GATConfig,
    GCNConfig,
    GraphSAGEConfig,
    GINConfig,
    PredictorConfig,
    TrainingConfig,
    MODEL_CONFIGS,
)

__all__ = [
    'BaseConfig',
    'GATConfig',
    'GCNConfig',
    'GraphSAGEConfig',
    'GINConfig',
    'PredictorConfig',
    'TrainingConfig',
    'MODEL_CONFIGS',
]
