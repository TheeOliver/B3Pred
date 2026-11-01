"""
Test MolCLR model trained on B3DB on S-data dataset
Loads saved model and scaler, evaluates on S-data
"""

import os
import sys
import json
import warnings
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

SDATA_PATH   = os.environ.get('MOLCLR_SDATA_PATH', '/manu/data')  # S-data location
OUTPUT_DIR   = os.environ.get('MOLCLR_OUTPUT_DIR', '/manu/results/test_sdata')
MODEL_DIR    = os.environ.get('MOLCLR_MODEL_DIR', '/manu/results/molclr_mydata')  # Directory with saved model

BATCH_SIZE   = 1024
NUM_WORKERS  = 4
THRESHOLD    = 0.5

# Automatically find the latest model
import glob
model_files = glob.glob(os.path.join(MODEL_DIR, 'molclr_model_*.pt'))
if not model_files:
    raise FileNotFoundError(f"No model found in {MODEL_DIR}")
MODEL_FILE = max(model_files, key=os.path.getctime)

print(f"Using model: {MODEL_FILE}")


def load_sdata():
    """Load S-data dataset."""
    print(f"\nLoading S-data from: {SDATA_PATH}")
    
    # Try different possible filenames
    possible_files = ['test.csv', 'sdata.csv', 'data.csv', 'S-data.csv', 'sdata_test.csv']
    test_file = None
    for filename in possible_files:
        candidate = os.path.join(SDATA_PATH, filename)
        if os.path.exists(candidate):
            test_file = candidate
            break
    
    if test_file is None:
        raise FileNotFoundError(f"S-data not found in {SDATA_PATH}. Tried: {possible_files}")
    
    print(f"Found S-data file: {test_file}")
    df = pd.read_csv(test_file)

    # Accept either the Drug/Y convention or this project's SMILES/BBB convention
    if 'Drug' in df.columns and 'Y' in df.columns:
        smiles_col, label_col = 'Drug', 'Y'
    elif 'SMILES' in df.columns and 'BBB' in df.columns:
        smiles_col, label_col = 'SMILES', 'BBB'
    elif 'SMILES' in df.columns and 'target' in df.columns:
        smiles_col, label_col = 'SMILES', 'target'
    else:
        raise ValueError(f"S-data missing expected columns. Found: {list(df.columns)}")

    smiles = df[smiles_col].tolist()
    labels = df[label_col].astype(float).tolist()
    
    # Remove any NaN values
    valid_data = [(s, l) for s, l in zip(smiles, labels) if not np.isnan(l)]
    smiles, labels = zip(*valid_data) if valid_data else ([], [])
    smiles = list(smiles)
    labels = list(labels)
    
    print(f"\n📊 S-data Statistics:")
    print(f"  Total molecules: {len(smiles)}")
    
    if labels:
        n_pos = sum(1 for y in labels if y > 0.5)
        n_neg = len(labels) - n_pos
        print(f"  Positive (1): {n_pos} ({100*n_pos/len(labels):.1f}%)")
        print(f"  Negative (0): {n_neg} ({100*n_neg/len(labels):.1f}%)")
    
    return smiles, labels


def _compute_single_fp(args):
    """Worker function for parallel fingerprint computation."""
    idx, smiles, n_bits, radius = args
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, idx
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, idx


def compute_morgan_fingerprints(smiles_list, n_bits=2048, radius=2):
    """Compute Morgan fingerprints in parallel across CPU cores."""
    import multiprocessing
    n_workers = min(multiprocessing.cpu_count(), 8)
    args = [(idx, smi, n_bits, radius) for idx, smi in enumerate(smiles_list)]
    
    fingerprints = []
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
    valid_indices = []
    
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i+batch_size]
        batch_valid = []
        batch_mols = []
        
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
        layers = []
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
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'embedding': self.embeddings[idx], 'label': self.labels[idx]}


def compute_all_metrics(y_true, y_pred, y_prob):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': (tp/(tp+fn) + tn/(tn+fp)) / 2 if (tp+fn) > 0 and (tn+fp) > 0 else 0,
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'auc_roc': roc_auc_score(y_true, y_prob),
        'auc_pr': average_precision_score(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'brier_score': brier_score_loss(y_true, y_prob),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'n_samples': len(y_true),
        'n_positive': int(tp + fn),
        'n_negative': int(tn + fp),
    }


def evaluate_model(model, loader, device, threshold=0.5, return_preds=False):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embedding'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(embeddings).squeeze()
            probs = torch.sigmoid(outputs.float())
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > threshold).astype(int)
    
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)
    if return_preds:
        metrics['predictions'] = all_preds.tolist()
        metrics['probabilities'] = all_probs.tolist()
        metrics['labels'] = all_labels.tolist()
    return metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "MolCLR" if MOLCLR_AVAILABLE else "Morgan-FP (fallback)"
    
    print(f"\n{'='*70}")
    print(f"{model_name} TESTING ON S-DATA")
    print(f"{'='*70}")
    print(f"Device      : {device}")
    print(f"Model file  : {MODEL_FILE}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Load S-data
    smiles, labels = load_sdata()
    
    if len(smiles) == 0:
        raise ValueError("No valid molecules found in S-data")
    
    # Load checkpoint
    print("\n🔧 Loading model and scaler...")
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    
    # Extract features from S-data
    print(f"\n🔬 Extracting {model_name} representations...")
    if MOLCLR_AVAILABLE:
        embeddings, valid_indices = extract_molclr_embeddings(smiles, device)
    else:
        print("  Using Morgan fingerprints (2048 bits, radius=2)")
        embeddings, valid_indices = compute_morgan_fingerprints(smiles)
    
    # Filter labels to match valid molecules
    valid_labels = [labels[i] for i in valid_indices]
    
    print(f"\n  Valid molecules: {len(embeddings)}/{len(smiles)}")
    
    # Apply the same scaler that was used during training
    scaler = checkpoint['scaler']
    embeddings_scaled = scaler.transform(embeddings)
    
    # Create model
    input_dim = checkpoint['config']['input_dim']
    hidden_dims = checkpoint['config']['hidden_dims']
    model = SimpleMLP(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded (best val AUC on B3DB: {checkpoint['best_val_auc']:.4f})")
    
    # Create dataset and loader
    dataset = EmbeddingDataset(embeddings_scaled, valid_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS, pin_memory=True)
    
    # Evaluate
    print("\n📊 Evaluating on S-data...")
    metrics = evaluate_model(model, loader, device, return_preds=True)
    
    # Print results
    print(f"\n📈 Results on S-data:")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  Specificity:    {metrics['specificity']:.4f}")
    print(f"  F1-Score:       {metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:        {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:         {metrics['auc_pr']:.4f}")
    print(f"  MCC:            {metrics['mcc']:.4f}")
    print(f"\n📊 Confusion Matrix:")
    print(f"  TP={metrics['true_positives']}  TN={metrics['true_negatives']}  "
          f"FP={metrics['false_positives']}  FN={metrics['false_negatives']}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(OUTPUT_DIR, f'molclr_sdata_results_{timestamp}.json')
    preds_file = os.path.join(OUTPUT_DIR, f'molclr_sdata_predictions_{timestamp}.csv')
    
    with open(result_file, 'w') as f:
        json.dump({
            'dataset': 'S-data',
            'model': model_name,
            'model_trained_on': 'b3db_tanimoto',
            'model_file': MODEL_FILE,
            'best_val_auc_on_b3db': checkpoint['best_val_auc'],
            'test_metrics': {k: v for k, v in metrics.items() 
                            if k not in ('predictions', 'probabilities', 'labels')},
        }, f, indent=2)
    
    pd.DataFrame({
        'true_label': metrics['labels'],
        'predicted_prob': metrics['probabilities'],
        'predicted_class': metrics['predictions'],
    }).to_csv(preds_file, index=False)
    
    print(f"\n✅ Results saved to:")
    print(f"  {result_file}")
    print(f"  {preds_file}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)