"""Traditional ML baseline models for BBB prediction."""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
)


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Convert SMILES string to Morgan fingerprint.

    Args:
        smiles: SMILES string
        radius: Radius for Morgan fingerprint
        n_bits: Number of bits in fingerprint

    Returns:
        Numpy array of fingerprint or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def build_feature_matrix(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    target_col: str = "target",
    n_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix from SMILES strings.

    Args:
        df: DataFrame with SMILES and target columns
        smiles_col: Name of SMILES column
        target_col: Name of target column
        n_bits: Number of bits for fingerprint

    Returns:
        Tuple of (features array, targets array)
    """
    X, y = [], []
    skipped = 0

    for i, row in df.iterrows():
        fp = smiles_to_morgan_fp(row[smiles_col], n_bits=n_bits)
        if fp is not None:
            X.append(fp)
            y.append(row[target_col])
        else:
            skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} invalid SMILES")

    return np.array(X), np.array(y)


def evaluate_baseline(
    model,
    X: np.ndarray,
    y: np.ndarray,
    name: str = "Dataset"
) -> Dict[str, float]:
    """
    Evaluate baseline model with comprehensive metrics.

    Args:
        model: Trained sklearn model
        X: Feature matrix
        y: Target labels
        name: Name of dataset

    Returns:
        Dictionary of metrics including accuracy, precision, specificity,
        sensitivity, recall, f1, auc, and MCC
    """
    y_pred = model.predict(X)

    # Try to get probabilities for AUC
    y_proba = None
    try:
        y_proba = model.predict_proba(X)
    except:
        pass

    # Calculate confusion matrix for specificity and sensitivity
    # Assumes binary classification with classes [0, 1]
    cm = confusion_matrix(y, y_pred)

    # Handle both binary and multi-class cases
    if cm.shape == (2, 2):
        # Binary classification
        tn, fp, fn, tp = cm.ravel()

        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Sensitivity = TP / (TP + FN) = Recall
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        # Multi-class: calculate macro-averaged specificity and sensitivity
        specificity_list = []
        sensitivity_list = []

        for i in range(cm.shape[0]):
            # For class i
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp

            spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sens_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            specificity_list.append(spec_i)
            sensitivity_list.append(sens_i)

        specificity = np.mean(specificity_list)
        sensitivity = np.mean(sensitivity_list)

    # Calculate standard metrics
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "specificity": specificity,
        "sensitivity": sensitivity,
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y, y_pred),
    }

    # Calculate AUC if probabilities available
    if y_proba is not None:
        try:
            if len(np.unique(y)) == 2:
                metrics["auc"] = roc_auc_score(y, y_proba[:, 1])
            else:
                metrics["auc"] = roc_auc_score(y, y_proba, multi_class="ovr")
        except Exception as e:
            metrics["auc"] = None
    else:
        metrics["auc"] = None

    # Print results
    print(f"\n{name} Results:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  F1-score:    {metrics['f1']:.4f}")
    if metrics['auc'] is not None:
        print(f"  AUC:         {metrics['auc']:.4f}")
    print(f"  MCC:         {metrics['mcc']:.4f}")

    return metrics


def get_logistic_regression():
    """Get logistic regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])


def get_random_forest():
    """Get random forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )


def get_svm():
    """Get SVM classifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ])