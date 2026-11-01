"""
Classical ML baselines for BBB permeation prediction.
Models: Logistic Regression, Random Forest, XGBoost
Features: ECFP (Morgan fingerprints, radius=2, 1024 bits)
"""

import os
import sys
import argparse
import warnings
import time
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not installed — skipping XGBoost model.")
    print("          To install: pip install xgboost --user")


# ──────────────────────────────────────────────────────────────────────────────
# Featurization
# ──────────────────────────────────────────────────────────────────────────────

def smiles_to_ecfp(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp)


def featurize(df, smiles_col, label_col, radius=2, n_bits=1024):
    """Convert a dataframe to (X, y), skipping invalid SMILES."""
    X, y, kept = [], [], []
    n_invalid = 0
    for _, row in df.iterrows():
        fp = smiles_to_ecfp(str(row[smiles_col]), radius, n_bits)
        if fp is None:
            n_invalid += 1
            continue
        X.append(fp)
        y.append(int(row[label_col]))
        kept.append(row[smiles_col])
    if n_invalid:
        print(f"  [INFO] Skipped {n_invalid} invalid SMILES")
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path, smiles_col=None, label_col=None):
    df = pd.read_csv(path)
    # Auto-detect columns if not provided
    if smiles_col is None:
        for c in ["SMILES", "smiles", "Smiles"]:
            if c in df.columns:
                smiles_col = c
                break
        else:
            raise ValueError(f"No SMILES column found in {path}. Columns: {list(df.columns)}")
    if label_col is None:
        for c in ["target", "BBB", "label", "Y"]:
            if c in df.columns:
                label_col = c
                break
        else:
            raise ValueError(f"No label column found in {path}. Columns: {list(df.columns)}")
    return df, smiles_col, label_col


def load_dataset(data_dir, prefix, sdata_path, radius=2, n_bits=1024):
    """Load train+val (merged) and test splits, plus sdata external test."""
    train_path = os.path.join(data_dir, f"{prefix}_tanimoto_train.csv")
    val_path   = os.path.join(data_dir, f"{prefix}_tanimoto_val.csv")
    test_path  = os.path.join(data_dir, f"{prefix}_tanimoto_test.csv")

    print(f"\n  Loading {prefix} train: {train_path}")
    train_df, sc, lc = load_csv(train_path)
    print(f"  Loading {prefix} val:   {val_path}")
    val_df, _, _ = load_csv(val_path, smiles_col=sc, label_col=lc)
    print(f"  Loading {prefix} test:  {test_path}")
    test_df, _, _ = load_csv(test_path, smiles_col=sc, label_col=lc)

    # Merge train + val (classical ML doesn't need a separate val)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"\n  Featurizing (ECFP radius={radius}, bits={n_bits})...")
    print(f"    train+val ({len(trainval_df)} molecules)")
    X_train, y_train = featurize(trainval_df, sc, lc, radius, n_bits)
    print(f"    test      ({len(test_df)} molecules)")
    X_test,  y_test  = featurize(test_df, sc, lc, radius, n_bits)

    print(f"\n  Loading sdata: {sdata_path}")
    sdata_df, s_sc, s_lc = load_csv(sdata_path)
    print(f"    sdata ({len(sdata_df)} molecules)")
    X_sdata, y_sdata = featurize(sdata_df, s_sc, s_lc, radius, n_bits)

    print(f"\n  Shapes — X_train: {X_train.shape}, X_test: {X_test.shape}, X_sdata: {X_sdata.shape}")
    class_counts = {v: int((y_train == v).sum()) for v in [0, 1]}
    print(f"  Class balance (train+val): {class_counts}")

    return X_train, y_train, X_test, y_test, X_sdata, y_sdata


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "Acc":  round(accuracy_score(y_true, y_pred), 4),
        "Prec": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Rec":  round(recall_score(y_true, y_pred, zero_division=0), 4),
        "Spec": round(specificity, 4),
        "F1":   round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUC":  round(roc_auc_score(y_true, y_prob), 4),
        "MCC":  round(matthews_corrcoef(y_true, y_pred), 4),
    }


def print_metrics(split_name, metrics):
    print(f"\n    [{split_name}]")
    header = "  ".join(f"{k:>6}" for k in metrics)
    values = "  ".join(f"{v:>6}" for v in metrics.values())
    print(f"      {header}")
    print(f"      {values}")


# ──────────────────────────────────────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────────────────────────────────────

def get_models(seed=42):
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            criterion="entropy",
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
            oob_score=True,
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def run(dataset_name, data_dir, sdata_path, results_dir, radius, n_bits, seed=42):
    print("=" * 60)
    print(f"  Dataset: {dataset_name.upper()}")
    print("=" * 60)

    X_train, y_train, X_test, y_test, X_sdata, y_sdata = load_dataset(
        data_dir, dataset_name, sdata_path, radius, n_bits
    )

    os.makedirs(results_dir, exist_ok=True)
    all_rows = []

    for model_name, model in get_models(seed=seed).items():
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name}")
        print(f"{'─'*60}")

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"  Training done in {elapsed:.1f}s")

        dataset_tag = dataset_name.replace("moleculenet", "molnet")
        for split_name, X_eval, y_eval in [
            (f"train_{dataset_tag}",  X_train, y_train),
            (f"test_{dataset_tag}",   X_test,  y_test),
            (f"sdata_{dataset_tag}",  X_sdata, y_sdata),
        ]:
            y_prob = model.predict_proba(X_eval)[:, 1]
            metrics = compute_metrics(y_eval, y_prob)
            print_metrics(split_name, metrics)
            row = {
                "dataset":  dataset_name,
                "model":    model_name,
                "split":    split_name,
                "n_samples": len(y_eval),
                **metrics,
            }
            all_rows.append(row)

        if model_name == "RandomForest":
            print(f"\n    OOB score: {model.oob_score_:.4f}")

    results_df = pd.DataFrame(all_rows)
    out_path = os.path.join(results_dir, f"{dataset_name}_classical_ml_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n  Results saved to: {out_path}")

    # Print a clean summary table
    dataset_tag = dataset_name.replace("moleculenet", "molnet")
    print(f"\n{'='*70}")
    print(f"  SUMMARY — {dataset_name.upper()}")
    print(f"  {'Model':<22}  {'Split':<18}  {'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'Spec':>5}  {'F1':>5}  {'AUC':>5}  {'MCC':>6}")
    print(f"{'='*70}")
    for model_name in results_df["model"].unique():
        for split in [f"test_{dataset_tag}", f"sdata_{dataset_tag}"]:
            row = results_df[(results_df["model"] == model_name) & (results_df["split"] == split)]
            if len(row):
                r = row.iloc[0]
                print(f"  {model_name:<22}  {split:<18}  "
                      f"{r['Acc']:>5}  {r['Prec']:>5}  {r['Rec']:>5}  "
                      f"{r['Spec']:>5}  {r['F1']:>5}  {r['AUC']:>5}  {r['MCC']:>6}")

    return results_df


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Classical ML baselines for BBB prediction")
    parser.add_argument("--dataset", choices=["b3db", "molnet", "both"], default="both",
                        help="Which training dataset to use")
    parser.add_argument("--data_dir", default="data",
                        help="Directory containing the tanimoto-split CSVs")
    parser.add_argument("--sdata_path", default="data/sdata.csv",
                        help="Path to sdata.csv (external test set)")
    parser.add_argument("--results_dir", default="experiments/classical_ml",
                        help="Directory to write result CSVs")
    parser.add_argument("--radius", type=int, default=2,
                        help="Morgan fingerprint radius (default: 2)")
    parser.add_argument("--n_bits", type=int, default=1024,
                        help="Morgan fingerprint bit length (default: 1024)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for all models (default: 42)")
    args = parser.parse_args()

    # b3db files are named "b3db_tanimoto_*.csv"
    dataset_map = {
        "b3db":   "b3db",
        "molnet": "moleculenet",
    }

    datasets_to_run = ["b3db", "molnet"] if args.dataset == "both" else [args.dataset]

    all_results = []
    for ds in datasets_to_run:
        prefix = dataset_map[ds]
        df = run(prefix, args.data_dir, args.sdata_path, args.results_dir, args.radius, args.n_bits,
                 seed=args.seed)
        all_results.append(df)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.results_dir, "all_datasets_classical_ml_results.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n  Combined results saved to: {combined_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
