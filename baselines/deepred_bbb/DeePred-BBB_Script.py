#!/usr/bin/env python3

"""
BBB Permeability prediction using a Deep Neural Network
as published in https://doi.org/10.3389/fnins.2022.858126
Edited by Dr. Freddy Bernal
Extended with evaluation metrics: Acc, Prec, Rec, Spec, F1, AUC, MCC
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix)
from keras.models import load_model


def bbb(input_bbb: pd.DataFrame) -> tuple:
    train_data = pd.read_csv(os.path.join(path, "data.csv"))
    train_data = train_data.drop(columns=[c for c in train_data.columns if not np.issubdtype(train_data[c].dtype, np.number)])

    scaler = StandardScaler()
    scaler.fit(train_data)

    # PaDEL occasionally emits Infinity for some descriptor columns (e.g. BCUT-type
    # descriptors on small/degenerate molecules); StandardScaler tolerates NaN but not inf.
    input_bbb = input_bbb.replace([np.inf, -np.inf], np.nan)

    bbb_user_input = scaler.transform(input_bbb.to_numpy())

    loaded_model = load_model(os.path.join(path, "DeePredmodel.h5"))
    print("Model loaded")

    raw = loaded_model.predict(bbb_user_input)  # (n, 2); model input is 2D (None, 1917)
    probabilities = raw[:, 1]                   # BBB+ probability

    nan_mask = np.isnan(probabilities)
    if nan_mask.any():
        print(f"Warning: {nan_mask.sum()} NaN predictions replaced with 0.5")
        probabilities = np.where(nan_mask, 0.5, probabilities)

    predictions = probabilities.round().astype(int)
    return predictions, probabilities

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute Acc, Prec, Rec, Spec, F1, AUC, MCC."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        'Accuracy':    accuracy_score(y_true, y_pred),
        'Precision':   precision_score(y_true, y_pred, zero_division=0),
        'Recall':      recall_score(y_true, y_pred, zero_division=0),
        'Specificity': specificity,
        'F1':          f1_score(y_true, y_pred, zero_division=0),
        'AUC':         roc_auc_score(y_true, y_prob),
        'MCC':         matthews_corrcoef(y_true, y_pred),
    }


def run_padel(folder: str, features_path: str) -> None:
    """Run PaDEL-Descriptor, working around its known JVM-hang-after-completion bug.

    PaDEL's descriptor calculation finishes processing every molecule but the JVM
    doesn't call System.exit() afterwards (its AWT/thread-pool machinery keeps
    non-daemon threads alive, especially with no display in a container), so
    subprocess.call() blocks forever waiting on a process that will never exit.
    '-Djava.awt.headless=true' avoids the AWT side of that. As a backstop, we also
    poll the output CSV and kill the process once it stops growing.
    """
    if os.path.exists(features_path):
        os.remove(features_path)

    padel_cmd = [
        'java', '-Djava.awt.headless=true', '-jar',
        os.path.join(path, 'PaDEL-Descriptor/PaDEL-Descriptor.jar'),
        '-descriptortypes',
        os.path.join(path, 'PaDEL-Descriptor/descriptors.xml'),
        '-dir', folder, '-file', features_path,
        '-2d', '-fingerprints', '-removesalt', '-detectaromaticity',
        '-standardizenitro']

    proc = subprocess.Popen(padel_cmd)

    poll_interval = 5       # seconds between checks
    stable_checks_needed = 3  # consecutive unchanged polls before treating output as final
    hard_timeout = 3600     # 1h safety net; normal runs finish in minutes

    stable_count = 0
    last_n_rows = -1
    elapsed = 0

    while True:
        ret = proc.poll()
        if ret is not None:
            break

        n_rows = 0
        if os.path.exists(features_path):
            with open(features_path) as f:
                n_rows = sum(1 for _ in f) - 1  # minus header

        if n_rows > 0 and n_rows == last_n_rows:
            stable_count += 1
        else:
            stable_count = 0
        last_n_rows = n_rows

        if stable_count >= stable_checks_needed:
            print(f"PaDEL output stable at {n_rows} rows but the process did not "
                  f"exit on its own (known PaDEL/JVM hang); terminating it.")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            break

        time.sleep(poll_interval)
        elapsed += poll_interval
        if elapsed >= hard_timeout:
            print(f"PaDEL did not finish within {hard_timeout}s; killing it.")
            proc.kill()
            proc.wait()
            break


def run_prediction(folder: str, labels: pd.Series) -> None:
    """Calculate descriptors and predict BBB permeability, then evaluate."""
    features_path = os.path.join(folder, 'PaDEL_features.csv')
    run_padel(folder, features_path)
    print("Features calculated")

    input_bbb = pd.read_csv(features_path)
    names = input_bbb['Name'].copy()
    input_bbb = input_bbb.drop(['Name'], axis=1)

    pred, prob = bbb(input_bbb)

    # Save predictions
    res = pd.DataFrame(names)
    res['Predicted_class'] = pred
    res['Probability'] = prob
    predictions_path = os.path.join(folder, 'DeePred-BBB_predictions.csv')
    res.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")

    # PaDEL names rows "AUTOGEN_smiles_<1-based line in smiles.smi>", skips molecules it
    # fails to process, and returns rows out of order (multithreaded). Align labels to
    # pred/prob by that index rather than assuming positional correspondence.
    row_idx = names.str.extract(r'(\d+)$')[0].astype(int) - 1
    n_total = len(labels)
    if len(row_idx) != n_total:
        print(f"Warning: PaDEL produced descriptors for {len(row_idx)}/{n_total} molecules; "
              f"{n_total - len(row_idx)} were skipped and excluded from evaluation.")
    y_true = labels.reset_index(drop=True).iloc[row_idx.values].values
    metrics = compute_metrics(y_true, pred, prob)

    print("\n========== Evaluation Metrics ==========")
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("========================================")

    # Save metrics to CSV too
    metrics_path = os.path.join(folder, 'DeePred-BBB_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        path = sys.argv[1]
        csv_path = sys.argv[2]
    elif len(sys.argv) == 2:
        path = sys.argv[1]
        csv_path = os.path.join(path, "sdata.csv")
    else:
        path = os.getcwd()
        csv_path = os.path.join(path, "sdata.csv")

    # Load CSV with SMILES and labels
    df = pd.read_csv(csv_path)

    # Adjust these column names to match your CSV
    smiles_col = 'SMILES'
    label_col  = 'BBB'

    if smiles_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must have '{smiles_col}' and '{label_col}' columns. "
                         f"Found: {list(df.columns)}")

    # Write .smi file for PaDEL
    smi_path = os.path.join(path, "smiles.smi")
    df[[smiles_col]].to_csv(smi_path, index=False, header=False)
    print(f".smi file written to {smi_path}")

    run_prediction(path, df[label_col])