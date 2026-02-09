"""
B3Pred - Blood-Brain Barrier Permeability Prediction using Graph Neural Networks
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Make key components easily accessible
from configs.base_config import BaseSettings
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset

__all__ = [
    'BaseSettings',
    'MoleculeDataset',
    'compute_feature_stats',
    'normalize_dataset',
]