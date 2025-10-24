import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sympy.printing.pytorch import torch


def test_model(loader, model, target_labels, hetero=False):
    model.eval()

    all_preds = []
    all_truths = []
    all_probs = []  # probability of positive class (class=1)

    with torch.no_grad():
        for data in loader:
            out = model(data)  # shape: [num_graphs, 2] for binary
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            truths = data.y.cpu().numpy()

            all_preds.extend(preds)
            all_truths.extend(truths)
            all_probs.extend(probs[:, 1])  # positive class (1) probability

    # Wrap in DataFrames
    pred_df = pd.DataFrame({target_labels[0]: all_preds})
    gt_df = pd.DataFrame({target_labels[0]: all_truths})

    res = dict()
    c = target_labels[0]

    # Standard metrics
    res[f'acc_{c}'] = accuracy_score(gt_df[c], pred_df[c])
    res[f'prec_{c}'] = precision_score(gt_df[c], pred_df[c], average='weighted', zero_division=0)
    res[f'recall_{c}'] = recall_score(gt_df[c], pred_df[c], average='weighted', zero_division=0)
    res[f'f1_{c}'] = f1_score(gt_df[c], pred_df[c], average='weighted', zero_division=0)
    res['macro_f1'] = res[f'f1_{c}']

    # AUC (binary)
    try:
        res[f'auc_{c}'] = roc_auc_score(gt_df[c], all_probs)
    except ValueError:
        res[f'auc_{c}'] = None  # if test set has only one class

    return res



