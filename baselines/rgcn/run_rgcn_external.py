"""
Adapted RGCN for external BBB datasets.

The original BBB-Penetration-Prediction model uses a fixed drug-protein interaction
graph (graph.pt) that cannot be rebuilt for arbitrary SMILES. This script builds a
drug-drug similarity graph using Mordred descriptors and Tanimoto similarity edges,
then trains the same RGCN architecture.

Usage:
    python run_rgcn_external.py \
        --train data/b3db_tanimoto_train.csv \
        --test  data/b3db_tanimoto_test.csv  \
        --smiles_col SMILES --label_col target \
        --out results/rgcn_b3db/metrics.json
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from pytorch_metric_learning import losses, distances, reducers
from pcgrad import PCGrad
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_morgan_fps(smiles_list, radius=2, n_bits=2048):
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return fps


def compute_mordred_features(smiles_list):
    from mordred import Calculator, descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
            valid_idx.append(i)
    df = calc.pandas(mols)
    # Coerce to numeric first so error objects become NaN, then drop all-NaN cols
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").fillna(0)
    feat = df.values.astype(np.float32)
    out = np.zeros((len(smiles_list), feat.shape[1]), dtype=np.float32)
    for new_i, orig_i in enumerate(valid_idx):
        out[orig_i] = feat[new_i]
    return out, valid_idx


def build_similarity_edges(fps, thresh_low=0.4, thresh_high=0.65):
    """
    Two relation types:
      0 = moderate similarity (thresh_low <= Tanimoto < thresh_high)
      1 = high similarity    (Tanimoto >= thresh_high)
    """
    n = len(fps)
    src, dst, etype = [], [], []
    for i in range(n):
        if fps[i] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], [fps[j] for j in range(i + 1, n) if fps[j] is not None])
        j_valid = [j for j in range(i + 1, n) if fps[j] is not None]
        for sim, j in zip(sims, j_valid):
            if sim >= thresh_high:
                src += [i, j]; dst += [j, i]; etype += [1, 1]
            elif sim >= thresh_low:
                src += [i, j]; dst += [j, i]; etype += [0, 0]
    if not src:
        # fallback: no edges → add self-loops with relation 0
        src = list(range(n)); dst = list(range(n)); etype = [0] * n
    return src, dst, etype


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

NUM_RELATIONS = 2


class Net(torch.nn.Module):
    def __init__(self, input_dim, dim1=512, dim2=256, dim3=128, dropout=0.645):
        super().__init__()
        self.conv1 = RGCNConv(input_dim, dim1, NUM_RELATIONS)
        self.conv2 = RGCNConv(dim1, dim2, NUM_RELATIONS)
        self.lin1 = Linear(dim2, dim3)
        self.lin2 = Linear(dim3, 2)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        emb = x
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), emb


# ---------------------------------------------------------------------------
# Training / eval
# ---------------------------------------------------------------------------

def train_step(model, optimizer, data, train_idx, loss_triplet):
    model.train()
    optimizer.zero_grad()
    out, emb = model(data.x, data.edge_index, data.edge_type)
    loss_tri = loss_triplet(emb[train_idx], data.y[train_idx])
    loss_nll = F.nll_loss(out[train_idx], data.y[train_idx])
    optimizer.pc_backward([loss_tri, loss_nll])
    optimizer.step()


@torch.no_grad()
def evaluate(model, data, idx):
    model.eval()
    out, _ = model(data.x, data.edge_index, data.edge_type)
    probs = out.exp()[idx]
    preds = probs.argmax(dim=-1).cpu().numpy()
    probs_pos = probs[:, 1].cpu().numpy()
    labels = data.y[idx].cpu().numpy()
    auc = roc_auc_score(labels, probs_pos) if len(set(labels)) > 1 else 0.5
    return preds, probs_pos, labels, auc


def compute_metrics(labels, preds, probs_pos):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    spec = recall_score(labels, preds, pos_label=0, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs_pos) if len(set(labels)) > 1 else float('nan')
    mcc  = matthews_corrcoef(labels, preds)
    return dict(Acc=acc, Prec=prec, Rec=rec, Spec=spec, F1=f1, AUC=auc, MCC=mcc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",      required=True)
    parser.add_argument("--test",       required=True)
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--label_col",  default="target")
    parser.add_argument("--out",        required=True)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--epochs",     type=int, default=500)
    parser.add_argument("--patience",   type=int, default=30)
    parser.add_argument("--lr",         type=float, default=0.00195)
    parser.add_argument("--wd",         type=float, default=0.00738)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    df_train = pd.read_csv(args.train)
    df_test  = pd.read_csv(args.test)

    # Normalise column names
    for df in [df_train, df_test]:
        df.rename(columns={c: "SMILES" for c in df.columns if c.lower() in ("smiles", "smi")}, inplace=True)
        df.rename(columns={c: "target" for c in df.columns if c.lower() in ("target", "bbb", "label", "y")}, inplace=True)

    all_smiles = df_train["SMILES"].tolist() + df_test["SMILES"].tolist()
    all_labels = df_train["target"].tolist() + df_test["target"].tolist()
    n_train    = len(df_train)

    print(f"Train: {n_train}, Test: {len(df_test)}")
    print("Computing Mordred descriptors …")
    features, valid_idx = compute_mordred_features(all_smiles)

    scaler   = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    print("Computing Morgan fingerprints for similarity edges …")
    fps = compute_morgan_fps(all_smiles)

    print("Building drug-drug similarity graph …")
    src, dst, etype = build_similarity_edges(fps)

    x          = torch.tensor(features, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type  = torch.tensor(etype, dtype=torch.long)
    y          = torch.tensor(all_labels, dtype=torch.long)
    data       = Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y).to(device)

    train_idx = torch.arange(n_train, dtype=torch.long)
    test_idx  = torch.arange(n_train, n_train + len(df_test), dtype=torch.long)

    # Triplet loss setup (same as original rgcn.py)
    margin   = 0.274
    distance = distances.CosineSimilarity()
    reducer  = reducers.ThresholdReducer(low=margin)
    loss_tri = losses.TripletMarginLoss(margin=margin, distance=distance, reducer=reducer,
                                        triplets_per_anchor=60)

    input_dim = features.shape[1]
    print(f"Input feature dim: {input_dim}")

    # 10-fold CV on train set to select best hyperparams (same as original)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed)
    cv_aucs = []
    y_train_np = data.y[train_idx].cpu().numpy()

    print("Running 10-fold CV on train set …")
    for fold, (tr, vl) in enumerate(skf.split(torch.zeros(n_train), y_train_np)):
        model = Net(input_dim).to(device)
        opt   = PCGrad(torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd))
        best_auc  = 0
        no_improve = 0
        for epoch in range(args.epochs):
            train_step(model, opt, data, train_idx[tr], loss_tri)
            _, pp, lb, auc = evaluate(model, data, train_idx[vl])
            if auc > best_auc:
                best_auc   = auc
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= args.patience:
                break
        cv_aucs.append(best_auc)
    print(f"CV AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")

    # Final model trained on full train set, early-stopped on random 15% val
    val_split  = int(0.15 * n_train)
    perm       = torch.randperm(n_train, generator=torch.Generator().manual_seed(args.seed))
    val_idx    = train_idx[perm[:val_split]]
    final_train_idx = train_idx[perm[val_split:]]

    print("Training final model on full train set …")
    model = Net(input_dim).to(device)
    opt   = PCGrad(torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd))
    best_auc   = 0
    no_improve  = 0
    best_state  = None
    for epoch in tqdm(range(args.epochs)):
        train_step(model, opt, data, final_train_idx, loss_tri)
        _, _, _, auc_val = evaluate(model, data, val_idx)
        if auc_val > best_auc:
            best_auc  = auc_val
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    preds, probs_pos, labels, _ = evaluate(model, data, test_idx)
    metrics = compute_metrics(labels, preds, probs_pos)

    print("\nTest results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
