"""
MolCLR / Morgan-FP for b3db_tanimoto
======================================
Binary classification task with pre-split train/val/test files.
METRICS: AUC-ROC, AUC-PR, F1, Accuracy, Precision, Recall,
         Specificity, MCC, Kappa, Log Loss, Brier Score, Confusion Matrix
"""

import os
import sys
import json
import warnings
from copy import deepcopy
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss, average_precision_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try to import MolCLR - falls back to Morgan fingerprints if unavailable
try:
    from adme_gnn.models.foundation import MolCLREncoder
    MOLCLR_AVAILABLE = True
except ImportError:
    MOLCLR_AVAILABLE = False
    print("⚠️  MolCLR not available, using Morgan fingerprint baseline instead")

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# ============== CONFIGURATION ==============

DATA_PATH  = os.environ.get('MOLCLR_DATA_PATH', '/manu/data/b3db_tanimoto')
OUTPUT_DIR = os.environ.get('MOLCLR_OUTPUT_DIR', '/manu/results/molclr_mydata')
DATASET_NAME = os.environ.get('MOLCLR_DATASET_NAME', 'b3db_tanimoto')
TASK_TYPE    = 'classification'

SEED         = int(os.environ.get('MOLCLR_SEED', 42))
BATCH_SIZE   = 1024  # A100-80GB: MLP on pre-computed embeddings is tiny, push batch size hard
NUM_WORKERS  = 4     # Parallel CPU workers to keep GPU fed
MAX_EPOCHS   = 100
PATIENCE     = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
HIDDEN_DIMS   = [256, 128]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_my_data():
    """Load pre-split dataset from /manu/data/b3db_tanimoto/{train,val,test}.csv"""
    print(f"\nLoading dataset from: {DATA_PATH}")

    train_file = os.path.join(DATA_PATH, 'train.csv')
    val_file   = os.path.join(DATA_PATH, 'val.csv')
    test_file  = os.path.join(DATA_PATH, 'test.csv')

    for f in [train_file, val_file, test_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    train_df = pd.read_csv(train_file)
    val_df   = pd.read_csv(val_file)
    test_df  = pd.read_csv(test_file)

    required_cols = ['Drug', 'Y']
    for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{name}.csv missing columns: {missing}")

    train_smiles = train_df['Drug'].tolist()
    train_labels = train_df['Y'].astype(float).tolist()
    val_smiles   = val_df['Drug'].tolist()
    val_labels   = val_df['Y'].astype(float).tolist()
    test_smiles  = test_df['Drug'].tolist()
    test_labels  = test_df['Y'].astype(float).tolist()

    print(f"\n📊 Dataset Statistics:")
    print(f"  Train: {len(train_smiles)} molecules")
    print(f"  Val:   {len(val_smiles)} molecules")
    print(f"  Test:  {len(test_smiles)} molecules")

    if train_labels:
        n_pos = sum(1 for y in train_labels if y > 0.5)
        n_neg = len(train_labels) - n_pos
        print(f"\n  Class distribution (train):")
        print(f"    Positive (1): {n_pos} ({100*n_pos/len(train_labels):.1f}%)")
        print(f"    Negative (0): {n_neg} ({100*n_neg/len(train_labels):.1f}%)")

    return {
        'train_smiles': train_smiles, 'train_labels': train_labels,
        'val_smiles':   val_smiles,   'val_labels':   val_labels,
        'test_smiles':  test_smiles,  'test_labels':  test_labels,
    }


def _compute_single_fp(args):
    """Worker function for parallel fingerprint computation."""
    idx, smiles, n_bits, radius = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, idx
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, idx


def compute_morgan_fingerprints(smiles_list, n_bits=2048, radius=2):
    """Compute Morgan fingerprints in parallel across CPU cores."""
    import multiprocessing
    n_workers = min(multiprocessing.cpu_count(), 8)
    args = [(idx, smi, n_bits, radius) for idx, smi in enumerate(smiles_list)]

    fingerprints  = []
    valid_indices = []

    with multiprocessing.Pool(n_workers) as pool:
        for arr, idx in pool.map(_compute_single_fp, args):
            if arr is not None:
                fingerprints.append(arr)
                valid_indices.append(idx)

    if not fingerprints:
        raise ValueError("No valid molecules found in dataset")

    return np.array(fingerprints), valid_indices


def extract_molclr_embeddings(smiles_list, device='cuda', batch_size=32):
    """Extract MolCLR embeddings for a list of SMILES."""
    if not MOLCLR_AVAILABLE:
        raise ImportError("MolCLR not available")

    encoder = MolCLREncoder(proj_dim=256).to(device)
    encoder.eval()

    all_embeddings = []
    valid_indices  = []

    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_valid  = []
        batch_mols   = []

        for idx, smiles in enumerate(batch_smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                batch_valid.append(idx)
                batch_mols.append(mol)

        if not batch_mols:
            continue

        with torch.no_grad():
            try:
                embeddings = encoder(batch_mols, device)
                all_embeddings.append(embeddings.cpu().numpy())
                valid_indices.extend([i + v for v in batch_valid])
            except Exception as e:
                print(f"    Warning: Batch failed: {e}")
                continue

    if not all_embeddings:
        raise ValueError("No valid embeddings could be extracted")

    return np.vstack(all_embeddings), valid_indices


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers   = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels     = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx], 'label': self.labels[idx]}


def train_epoch_mlp(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0
    n_samples  = 0

    for batch in loader:
        embeddings = batch['embedding'].to(device, non_blocking=True)
        labels     = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(embeddings).squeeze()
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        n_samples  += len(labels)

    return total_loss / n_samples


def compute_all_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        'accuracy':                  accuracy_score(y_true, y_pred),
        'balanced_accuracy':         (tp/(tp+fn) + tn/(tn+fp)) / 2,
        'precision':                 precision_score(y_true, y_pred, zero_division=0),
        'recall':                    recall_score(y_true, y_pred, zero_division=0),
        'specificity':               tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score':                  f1_score(y_true, y_pred, zero_division=0),
        'f2_score':                  (5*tp) / (5*tp + 4*fn + fp) if (5*tp + 4*fn + fp) > 0 else 0,
        'f0_5_score':                (1.25*tp) / (1.25*tp + 0.25*fn + fp) if (1.25*tp + 0.25*fn + fp) > 0 else 0,
        'true_positives':            int(tp),
        'true_negatives':            int(tn),
        'false_positives':           int(fp),
        'false_negatives':           int(fn),
        'tpr':                       tp / (tp + fn) if (tp + fn) > 0 else 0,
        'tnr':                       tn / (tn + fp) if (tn + fp) > 0 else 0,
        'fpr':                       fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr':                       fn / (fn + tp) if (fn + tp) > 0 else 0,
        'auc_roc':                   roc_auc_score(y_true, y_prob),
        'auc_pr':                    average_precision_score(y_true, y_prob),
        'log_loss':                  log_loss(y_true, y_prob),
        'brier_score':               brier_score_loss(y_true, y_prob),
        'mcc':                       matthews_corrcoef(y_true, y_pred),
        'kappa':                     cohen_kappa_score(y_true, y_pred),
        'youden_index':              (tp/(tp+fn) + tn/(tn+fp)) - 1,
        'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'false_discovery_rate':      fp / (tp + fp) if (tp + fp) > 0 else 0,
        'false_omission_rate':       fn / (tn + fn) if (tn + fn) > 0 else 0,
        'threat_score':              tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0,
        'n_samples':                 len(y_true),
        'n_positive':                int(tp + fn),
        'n_negative':                int(tn + fp),
    }


def evaluate_mlp(model, loader, device, threshold=0.5, return_preds=False):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embedding'].to(device, non_blocking=True)
            labels     = batch['label'].to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(embeddings).squeeze()
            probs = torch.sigmoid(outputs.float())

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = (all_probs > threshold).astype(int)

    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    if return_preds:
        metrics['predictions']   = all_preds.tolist()
        metrics['probabilities'] = all_probs.tolist()
        metrics['labels']        = all_labels.tolist()
    return metrics


def run_molclr_for_my_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device     = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "MolCLR" if MOLCLR_AVAILABLE else "Morgan-FP (fallback)"

    print(f"\n{'='*70}")
    print(f"{model_name} — b3db_tanimoto")
    print(f"{'='*70}")
    print(f"Device     : {device}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Hidden dims: {HIDDEN_DIMS}")
    print(f"Max epochs : {MAX_EPOCHS}  Patience: {PATIENCE}")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    set_seed(SEED)
    data = load_my_data()

    print(f"\n🔬 Extracting {model_name} representations...")
    if MOLCLR_AVAILABLE:
        train_embeddings, train_valid = extract_molclr_embeddings(data['train_smiles'], device)
        val_embeddings,   val_valid   = extract_molclr_embeddings(data['val_smiles'],   device)
        test_embeddings,  test_valid  = extract_molclr_embeddings(data['test_smiles'],  device)
    else:
        print("  Using Morgan fingerprints (2048 bits, radius=2)")
        train_embeddings, train_valid = compute_morgan_fingerprints(data['train_smiles'])
        val_embeddings,   val_valid   = compute_morgan_fingerprints(data['val_smiles'])
        test_embeddings,  test_valid  = compute_morgan_fingerprints(data['test_smiles'])

    train_labels = [data['train_labels'][i] for i in train_valid]
    val_labels   = [data['val_labels'][i]   for i in val_valid]
    test_labels  = [data['test_labels'][i]  for i in test_valid]

    print(f"\n  Valid molecules:")
    print(f"    Train: {len(train_embeddings)}/{len(data['train_smiles'])}")
    print(f"    Val:   {len(val_embeddings)}/{len(data['val_smiles'])}")
    print(f"    Test:  {len(test_embeddings)}/{len(data['test_smiles'])}")

    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings   = scaler.transform(val_embeddings)
    test_embeddings  = scaler.transform(test_embeddings)

    train_dataset = EmbeddingDataset(train_embeddings, train_labels)
    val_dataset   = EmbeddingDataset(val_embeddings,   val_labels)
    test_dataset  = EmbeddingDataset(test_embeddings,  test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    input_dim = train_embeddings.shape[1]
    model     = SimpleMLP(input_dim=input_dim, hidden_dims=HIDDEN_DIMS).to(device)


    print(f"\n📊 Model: input_dim={input_dim}  hidden={HIDDEN_DIMS}  "
          f"params={sum(p.numel() for p in model.parameters()):,}")

    n_pos = sum(1 for y in train_labels if y > 0.5)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 and n_neg > 0 \
                 else torch.tensor([1.0], device=device)
    print(f"📊 Class weighting: pos_weight = {pos_weight.item():.2f}")

    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    amp_scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    best_val_auc     = 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_metrics': []}

    print("\n🚀 Starting training...")
    print("-" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss  = train_epoch_mlp(model, train_loader, optimizer, criterion, device, amp_scaler)
        val_metrics = evaluate_mlp(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_metrics'].append({
            'epoch':    epoch,
            'auc_roc':  val_metrics['auc_roc'],
            'auc_pr':   val_metrics['auc_pr'],
            'f1_score': val_metrics['f1_score'],
            'accuracy': val_metrics['accuracy'],
            'log_loss': val_metrics['log_loss'],
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f} | "
                  f"AUC={val_metrics['auc_roc']:.4f} | "
                  f"AUCPR={val_metrics['auc_pr']:.4f} | "
                  f"F1={val_metrics['f1_score']:.4f}")

        if val_metrics['auc_roc'] > best_val_auc:
            best_val_auc     = val_metrics['auc_roc']
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
            print(f"  → New best model! (AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (val_AUC = {best_val_auc:.4f})")
        
        # ========== ADD THIS SECTION ==========
        # Save the trained model and scaler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(OUTPUT_DIR, f'molclr_model_{timestamp}.pt')
        
        torch.save({
            'model_state_dict': best_model_state,
            'config': {
                'input_dim': input_dim,
                'hidden_dims': HIDDEN_DIMS,
            },
            'scaler': scaler,  # Save the fitted scaler
            'best_val_auc': best_val_auc,
        }, model_path)
        print(f"💾 Model and scaler saved to: {model_path}")
        # ========== END ADDED SECTION ==========

    print("\n" + "=" * 70)
    print("📊 FINAL TEST SET EVALUATION")
    print("=" * 70)

    test_metrics = evaluate_mlp(model, test_loader, device, return_preds=True)

    print(f"\n📈 Threshold-based metrics (threshold=0.5):")
    print(f"  Accuracy:             {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy:    {test_metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:            {test_metrics['precision']:.4f}")
    print(f"  Recall (Sensitivity): {test_metrics['recall']:.4f}")
    print(f"  Specificity:          {test_metrics['specificity']:.4f}")
    print(f"  F1-Score:             {test_metrics['f1_score']:.4f}")
    print(f"  F2-Score:             {test_metrics['f2_score']:.4f}")
    print(f"  F0.5-Score:           {test_metrics['f0_5_score']:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"  TP={test_metrics['true_positives']}  TN={test_metrics['true_negatives']}  "
          f"FP={test_metrics['false_positives']}  FN={test_metrics['false_negatives']}")
    print(f"\n📉 Probability-based metrics:")
    print(f"  AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:      {test_metrics['auc_pr']:.4f}")
    print(f"  Log Loss:    {test_metrics['log_loss']:.4f}")
    print(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"\n🔗 Correlation metrics:")
    print(f"  MCC:         {test_metrics['mcc']:.4f}")
    print(f"  Cohen Kappa: {test_metrics['kappa']:.4f}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f'molclr_results_{timestamp}.json')
    preds_file  = os.path.join(OUTPUT_DIR, f'molclr_predictions_{timestamp}.csv')

    with open(result_file, 'w') as f:
        json.dump({
            'dataset':        DATASET_NAME,
            'model':          model_name,
            'task_type':      TASK_TYPE,
            'train_size':     len(train_embeddings),
            'val_size':       len(val_embeddings),
            'test_size':      len(test_embeddings),
            'best_val_auc':   best_val_auc,
            'epochs_trained': len(history['train_loss']),
            'config': {
                'batch_size':    BATCH_SIZE,
                'hidden_dims':   HIDDEN_DIMS,
                'learning_rate': LEARNING_RATE,
                'weight_decay':  WEIGHT_DECAY,
                'max_epochs':    MAX_EPOCHS,
                'patience':      PATIENCE,
                'seed':          SEED,
            },
            'val_history':  history['val_metrics'],
            'test_metrics': {k: v for k, v in test_metrics.items()
                             if k not in ('predictions', 'probabilities', 'labels')},
        }, f, indent=2)

    pd.DataFrame({
        'true_label':      test_metrics['labels'],
        'predicted_prob':  test_metrics['probabilities'],
        'predicted_class': test_metrics['predictions'],
    }).to_csv(preds_file, index=False)

    print(f"\n✅ Results saved to:")
    print(f"  {result_file}")
    print(f"  {preds_file}")


if __name__ == "__main__":
    try:
        run_molclr_for_my_data()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)