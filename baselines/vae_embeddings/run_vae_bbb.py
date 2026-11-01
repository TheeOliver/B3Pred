"""
VAE-Embeddings pipeline for external BBB datasets.

Pipeline:
  1. Prepare fold indices from train CSV
  2. [Optional] Generate PVAE embeddings if a pre-trained checkpoint is given
     (requires pvae/ directory and checkpoint).  If no checkpoint is found,
     falls back to Morgan fingerprint features (1024-bit ECFP4).
  3. Train the downstream ResNet20 (TF1 session-based) or sklearn MLP/LR.
  4. Evaluate on the held-out test CSV.
  5. Report Acc, Prec, Rec, Spec, F1, AUC, MCC.

Usage:
    # Using Morgan fingerprints (no pre-trained VAE needed):
    python run_vae_bbb.py \
        --train ../../data/b3db_tanimoto_train.csv \
        --test  ../../data/b3db_tanimoto_test.csv  \
        --out   ../../results/others/vae/b3db \
        --model MLP --feature fingerprint

    # Using PVAE embeddings (pre-trained checkpoint required):
    python run_vae_bbb.py \
        --train ... --test ... --out ... \
        --model MLP --feature vae_emb \
        --pvae_dir pvae/ --pvae_checkpoint E70.pytorch
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def morgan_fingerprints(smiles_list, radius=2, n_bits=1024):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(n_bits, dtype=np.float32))
        else:
            arr = np.zeros(n_bits, dtype=np.float32)
            bit_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(bit_vect, arr)
            fps.append(arr)
    return np.vstack(fps)


def pvae_embeddings(smiles_list, pvae_dir, checkpoint_name):
    """Generate embeddings using a pre-trained PVAE checkpoint."""
    import ast
    import torch
    sys.path.insert(0, pvae_dir)
    from params import VAEParams
    from model_property2 import PropertyVAE

    params = VAEParams()
    checkpoint_path = os.path.join(pvae_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"PVAE checkpoint not found: {checkpoint_path}")

    model = PropertyVAE(params)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    from data_prepration import smiles_to_tensor
    embeddings = []
    with torch.no_grad():
        for smi in smiles_list:
            try:
                x = smiles_to_tensor(smi, params)
                _, mu, _ = model.encode(x.unsqueeze(0))
                embeddings.append(mu.squeeze(0).numpy())
            except Exception:
                embeddings.append(np.zeros(params.latent_dim, dtype=np.float32))
    return np.vstack(embeddings)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    return dict(
        Acc  = accuracy_score(y_true, y_pred),
        Prec = precision_score(y_true, y_pred, zero_division=0),
        Rec  = recall_score(y_true, y_pred, zero_division=0),
        Spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        F1   = f1_score(y_true, y_pred, zero_division=0),
        AUC  = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
        MCC  = matthews_corrcoef(y_true, y_pred),
    )


# ---------------------------------------------------------------------------
# ResNet20 via TF1 (train_logBB_resnet equivalent, standalone)
# ---------------------------------------------------------------------------

def train_resnet_tf1(X_tr, y_tr, X_te, y_te, epochs=85, lr=1e-5, batch_size=47, l2_wd=1e-5):
    import tensorflow as tf
    from resnet import ResNet20

    tf.reset_default_graph()
    input_len = X_tr.shape[1]
    X_ph = tf.placeholder(tf.float32, [None, input_len])
    y_ph = tf.placeholder(tf.float32, [None, 1])

    model = ResNet20(input_len)
    preds = model.forward(X_ph)

    model_weights = [v for v in tf.trainable_variables() if "w_fc" in v.name or "kernel" in v.name]
    l2_loss = tf.reduce_sum([l2_wd * tf.nn.l2_loss(w) for w in model_weights])
    xent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_ph, logits=preds))
    loss = xent + l2_loss
    predicted = tf.nn.sigmoid(preds)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, epochs + 1):
            for i in range(0, len(X_tr), batch_size):
                xb = X_tr[i: i + batch_size]
                yb = y_tr[i: i + batch_size].reshape(-1, 1)
                sess.run(optimizer, feed_dict={X_ph: xb, y_ph: yb})

        probs = sess.run(predicted, feed_dict={X_ph: X_te}).flatten()

    y_pred = (probs >= 0.5).astype(int)
    return y_pred, probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",             required=True)
    parser.add_argument("--test",              required=True)
    parser.add_argument("--out",               required=True)
    parser.add_argument("--smiles_col",        default="SMILES")
    parser.add_argument("--label_col",         default="target")
    parser.add_argument("--model",             choices=["ResNet", "MLP", "LR"], default="MLP")
    parser.add_argument("--feature",           choices=["fingerprint", "vae_emb"], default="fingerprint")
    parser.add_argument("--pvae_dir",          default="pvae")
    parser.add_argument("--pvae_checkpoint",   default="E70.pytorch")
    parser.add_argument("--epochs",            type=int,   default=85)
    parser.add_argument("--lr",                type=float, default=1e-5)
    parser.add_argument("--batch_size",        type=int,   default=47)
    parser.add_argument("--l2_wd",             type=float, default=1e-5)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    df_tr = pd.read_csv(args.train)
    df_te = pd.read_csv(args.test)

    for df in [df_tr, df_te]:
        df.rename(columns={c: "SMILES" for c in df.columns if c.lower() in ("smiles","smi")}, inplace=True)
        df.rename(columns={c: "target" for c in df.columns if c.lower() in ("target","bbb","label","y")}, inplace=True)

    smiles_tr = df_tr["SMILES"].tolist()
    smiles_te = df_te["SMILES"].tolist()
    y_tr = df_tr["target"].values.astype(int)
    y_te = df_te["target"].values.astype(int)

    print(f"Train: {len(smiles_tr)}, Test: {len(smiles_te)}, Feature: {args.feature}, Model: {args.model}")

    # Feature extraction
    if args.feature == "vae_emb":
        pvae_dir = os.path.join(os.path.dirname(__file__), args.pvae_dir)
        print(f"Generating PVAE embeddings from {pvae_dir} …")
        X_tr = pvae_embeddings(smiles_tr, pvae_dir, args.pvae_checkpoint)
        X_te = pvae_embeddings(smiles_te, pvae_dir, args.pvae_checkpoint)
    else:
        print("Computing Morgan fingerprints (ECFP4, 1024 bits) …")
        X_tr = morgan_fingerprints(smiles_tr)
        X_te = morgan_fingerprints(smiles_te)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_te   = scaler.transform(X_te).astype(np.float32)
    X_tr   = np.nan_to_num(X_tr)
    X_te   = np.nan_to_num(X_te)

    # Training
    if args.model == "ResNet":
        print("Training ResNet20 (TF1) …")
        y_pred, y_prob = train_resnet_tf1(X_tr, y_tr, X_te, y_te,
                                          epochs=args.epochs, lr=args.lr,
                                          batch_size=args.batch_size, l2_wd=args.l2_wd)
    elif args.model == "MLP":
        print("Training MLPClassifier …")
        clf = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=1000,
                            random_state=args.seed, early_stopping=True,
                            validation_fraction=0.1)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]
    elif args.model == "LR":
        print("Training Logistic Regression …")
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

    m = compute_metrics(y_te, y_pred, y_prob)

    print("\nTest results:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"metrics_{args.model}_{args.feature}.json")
    with open(out_path, "w") as f:
        json.dump(m, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
