"""
Model Evaluation Module

Provides functions to evaluate GNN models on test/validation data.
Computes comprehensive classification metrics.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def test_model(
        loader,
        model: torch.nn.Module,
        target_labels: List[str],
        hetero: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on a data loader.

    Computes comprehensive classification metrics including:
    - Accuracy
    - Precision, Recall, F1 (weighted)
    - AUC-ROC
    - Matthews Correlation Coefficient (MCC)
    - Specificity

    Args:
        loader: PyTorch DataLoader
        model: Trained model
        target_labels: List of target label names
        hetero: Whether using heterogeneous graphs (not currently used)

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    all_preds = []
    all_truths = []
    all_probs = []  # Probability of positive class (class=1)

    with torch.no_grad():
        for data in loader:
            # Forward pass
            out = model(data)  # shape: [num_graphs, num_classes]

            # Get predictions and probabilities
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            truths = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_truths.extend(truths)
            all_probs.extend(probs[:, 1])  # Positive class (1) probability

    # Convert to DataFrames for consistency
    pred_df = pd.DataFrame({target_labels[0]: all_preds})
    gt_df = pd.DataFrame({target_labels[0]: all_truths})

    # Compute metrics
    results = {}
    label = target_labels[0]

    # Standard classification metrics
    results[f'acc_{label}'] = accuracy_score(gt_df[label], pred_df[label])
    results[f'prec_{label}'] = precision_score(
        gt_df[label], pred_df[label],
        average='weighted',
        zero_division=0
    )
    results[f'recall_{label}'] = recall_score(
        gt_df[label], pred_df[label],
        average='weighted',
        zero_division=0
    )
    results[f'f1_{label}'] = f1_score(
        gt_df[label], pred_df[label],
        average='weighted',
        zero_division=0
    )
    results[f'mcc_{label}'] = matthews_corrcoef(gt_df[label], pred_df[label])

    # Specificity (recall for negative class)
    results[f'spec_{label}'] = recall_score(
        gt_df[label], pred_df[label],
        pos_label=0,
        average='weighted',
        zero_division=0
    )

    # Macro F1 (for compatibility)
    results['macro_f1'] = results[f'f1_{label}']

    # AUC-ROC (binary classification)
    try:
        results[f'auc_{label}'] = roc_auc_score(gt_df[label], all_probs)
    except ValueError as e:
        logger.warning(f"Could not compute AUC: {e}")
        results[f'auc_{label}'] = None

    return results


def compute_confusion_matrix(
        loader,
        model: torch.nn.Module
) -> np.ndarray:
    """
    Compute confusion matrix for model predictions.

    Args:
        loader: PyTorch DataLoader
        model: Trained model

    Returns:
        Confusion matrix as numpy array
    """
    model.eval()

    all_preds = []
    all_truths = []

    with torch.no_grad():
        for data in loader:
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            truths = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_truths.extend(truths)

    return confusion_matrix(all_truths, all_preds)


def evaluate_per_class_metrics(
        loader,
        model: torch.nn.Module,
        class_names: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class metrics (precision, recall, F1).

    Useful for imbalanced datasets to see performance on each class.

    Args:
        loader: PyTorch DataLoader
        model: Trained model
        class_names: Optional list of class names

    Returns:
        Dictionary mapping class names to their metrics
    """
    model.eval()

    all_preds = []
    all_truths = []

    with torch.no_grad():
        for data in loader:
            out = model(data)
            preds = out.argmax(dim=1).cpu().numpy()
            truths = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_truths.extend(truths)

    all_preds = np.array(all_preds)
    all_truths = np.array(all_truths)

    # Get unique classes
    classes = np.unique(all_truths)
    if class_names is None:
        class_names = [f"Class {i}" for i in classes]

    # Compute per-class metrics
    per_class_results = {}

    for cls, name in zip(classes, class_names):
        per_class_results[name] = {
            'precision': precision_score(
                all_truths, all_preds,
                labels=[cls],
                average=None,
                zero_division=0
            )[0],
            'recall': recall_score(
                all_truths, all_preds,
                labels=[cls],
                average=None,
                zero_division=0
            )[0],
            'f1': f1_score(
                all_truths, all_preds,
                labels=[cls],
                average=None,
                zero_division=0
            )[0],
            'support': np.sum(all_truths == cls)
        }

    return per_class_results