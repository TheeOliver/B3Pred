"""Evaluation utilities for BBB predictor."""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device('cpu')
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Evaluate a model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on

    Returns:
        Tuple of (metrics dict, predictions array, ground truth array)
    """
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)

            # Forward pass
            logits = model(data)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # Calculate confusion matrix first (needed for several metrics)
    cm = confusion_matrix(y_true, y_pred)

    # Extract confusion matrix values for binary classification
    # cm = [[TN, FP],
    #       [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()

        # Calculate specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Calculate sensitivity (same as recall for binary classification)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        # Handle edge case where only one class is present
        specificity = None
        sensitivity = None

    # Calculate standard metrics (binary classification)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'sensitivity': sensitivity if sensitivity is not None else recall_score(y_true, y_pred, average='binary',
                                                                                zero_division=0),
        'specificity': specificity,
    }

    # Calculate AUC if possible
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = None  # Single class in y_true

    # Calculate Matthews Correlation Coefficient
    try:
        from sklearn.metrics import matthews_corrcoef
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    except ValueError:
        metrics['mcc'] = None

    # Store confusion matrix
    metrics['confusion_matrix'] = cm.tolist()

    return metrics, y_pred, y_true


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print evaluation metrics in a formatted way.

    Args:
        metrics: Dictionary of metric names to values
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{'=' * 50}")
    print(f"{dataset_name} Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(
        f"Specificity: {metrics['specificity']:.4f}" if metrics.get('specificity') is not None else "Specificity: N/A")
    print(f"F1-score:    {metrics['f1']:.4f}")

    if metrics.get('auc') is not None:
        print(f"AUC:         {metrics['auc']:.4f}")

    if metrics.get('mcc') is not None:
        print(f"MCC:         {metrics['mcc']:.4f}")

    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            print(f"  TN: {tn}, FP: {fp}")
            print(f"  FN: {fn}, TP: {tp}")

    print(f"{'=' * 50}\n")