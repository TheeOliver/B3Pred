"""
BBB-Models wrapper: trains RFC / GBC / LogReg with Mordred descriptors.

The pre-trained pickled models in this repo were fitted on a proprietary
descriptor matrix for 968 compounds.  Since those feature vectors are not
reproducible for arbitrary SMILES, we re-train the same three sklearn model
families from scratch using Mordred 2D descriptors.

Usage:
    python run_bbm.py \
        --train ../../data/b3db_tanimoto_train.csv \
        --test  ../../data/b3db_tanimoto_test.csv  \
        --smiles_col SMILES --label_col target \
        --out   ../../results/others/bbm/b3db

Reports metrics for all three models and writes a JSON with the RFC result
(consistently best in the original paper).
"""

import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)

warnings.filterwarnings("ignore")


def compute_mordred(smiles_list, cols=None):
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol); valid.append(i)
    df = calc.pandas(mols)
    # Coerce everything to numeric; non-computable descriptors become NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    if cols is None:
        # First call (train): drop columns that are entirely NaN, record survivors
        df = df.dropna(axis=1, how="all")
        cols = list(df.columns)
    else:
        # Subsequent calls (test/sdata): align to train columns exactly
        df = df.reindex(columns=cols)
    df = df.fillna(0)
    mat = np.zeros((len(smiles_list), len(cols)), dtype=np.float32)
    for new_i, orig_i in enumerate(valid):
        mat[orig_i] = df.values[new_i]
    return mat, cols


def metrics(y_true, y_pred, y_prob):
    return dict(
        Acc  = accuracy_score(y_true, y_pred),
        Prec = precision_score(y_true, y_pred, zero_division=0),
        Rec  = recall_score(y_true, y_pred, zero_division=0),
        Spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        F1   = f1_score(y_true, y_pred, zero_division=0),
        AUC  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        MCC  = matthews_corrcoef(y_true, y_pred),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--test",       required=True)
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--label_col",  default="target")
    parser.add_argument("--out",        required=True)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    df_tr = pd.read_csv(args.train)
    df_te = pd.read_csv(args.test)

    for df in [df_tr, df_te]:
        df.rename(columns={c: "SMILES" for c in df.columns if c.lower() in ("smiles","smi")}, inplace=True)
        df.rename(columns={c: "target" for c in df.columns if c.lower() in ("target","bbb","label","y")}, inplace=True)

    print(f"Train: {len(df_tr)}, Test: {len(df_te)}")
    print("Computing Mordred descriptors …")

    X_tr, cols = compute_mordred(df_tr["SMILES"].tolist())
    X_te, _    = compute_mordred(df_te["SMILES"].tolist(), cols=cols)
    y_tr = df_tr["target"].values
    y_te = df_te["target"].values

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    X_tr   = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
    X_te   = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

    models = {
        "RFC":    RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                         min_samples_leaf=1, random_state=args.seed, n_jobs=-1),
        "GBC":    GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                              learning_rate=0.1, random_state=args.seed),
        "LogReg": LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs",
                                     multi_class="auto", random_state=args.seed),
    }

    all_results = {}
    for name, clf in models.items():
        print(f"\nTraining {name} …")
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
        m = metrics(y_te, y_pred, y_prob)
        all_results[name] = m
        print(f"  {name}: Acc={m['Acc']:.4f}  F1={m['F1']:.4f}  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics_all.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # RFC is the primary result (consistent with the original paper)
    rfc_path = os.path.join(args.out, "metrics_rfc.json")
    with open(rfc_path, "w") as f:
        json.dump(all_results["RFC"], f, indent=2)

    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
