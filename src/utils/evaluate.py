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
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Calculate AUC if possible
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics['auc'] = None  # Single class in y_true
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics, y_pred, y_true


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metric names to values
        dataset_name: Name of the dataset being evaluated
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name} Evaluation Results")
    print(f"{'='*50}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    
    if metrics.get('auc') is not None:
        print(f"AUC:       {metrics['auc']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics['confusion_matrix'])
        print(cm)
    
    print(f"{'='*50}\n")
