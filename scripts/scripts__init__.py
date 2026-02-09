"""Training and evaluation scripts"""

from .train import train_model
from .evaluate import test_model, compute_confusion_matrix, evaluate_per_class_metrics

__all__ = [
    'train_model',
    'test_model',
    'compute_confusion_matrix',
    'evaluate_per_class_metrics',
]