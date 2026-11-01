"""
Test ChemBERTa model trained on B3DB on S-data dataset
Loads saved model and tokenizer, evaluates on S-data
"""

import os
import sys
import json
import warnings
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss, average_precision_score,
)

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============

SDATA_PATH   = os.environ.get('CHEMBERTA_SDATA_PATH', '/manu/data')  # S-data location
OUTPUT_DIR   = os.environ.get('CHEMBERTA_OUTPUT_DIR', '/manu/results/test_sdata')
MODEL_DIR    = os.environ.get('CHEMBERTA_MODEL_DIR', '/manu/results/chemberta_mydata')  # Directory with saved model and tokenizer

BATCH_SIZE   = 256
NUM_WORKERS  = 4
THRESHOLD    = 0.5

# Automatically find the latest model and tokenizer
import glob
model_files = glob.glob(os.path.join(MODEL_DIR, 'chemberta_model_*.pt'))
if not model_files:
    raise FileNotFoundError(f"No model found in {MODEL_DIR}")
MODEL_FILE = max(model_files, key=os.path.getctime)

# Find the matching tokenizer directory
tokenizer_dirs = glob.glob(os.path.join(MODEL_DIR, 'tokenizer_*'))
if not tokenizer_dirs:
    raise FileNotFoundError(f"No tokenizer found in {MODEL_DIR}")
TOKENIZER_DIR = max(tokenizer_dirs, key=os.path.getctime)

print(f"Using model: {MODEL_FILE}")
print(f"Using tokenizer: {TOKENIZER_DIR}")


class ChemBERTaFineTuned(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1",
                 unfreeze_layers=2, hidden_size=768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Freeze all encoder layers (we're just loading trained weights)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Same head architecture as in training
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)


class ChemBERTaDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles = smiles_list
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.smiles[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_sdata():
    """Load S-data dataset."""
    print(f"\nLoading S-data from: {SDATA_PATH}")
    
    # Try different possible filenames
    possible_files = ['sdata.csv', 'S-data.csv', 'test.csv', 'data.csv', 'sdata_test.csv']
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
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs.squeeze().float())
            
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
    
    print(f"\n{'='*70}")
    print("CHEMBERTA TESTING ON S-DATA")
    print(f"{'='*70}")
    print(f"Device      : {device}")
    print(f"Model file  : {MODEL_FILE}")
    print(f"Tokenizer   : {TOKENIZER_DIR}")
    print(f"Output dir  : {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    # Load S-data
    smiles, labels = load_sdata()
    
    if len(smiles) == 0:
        raise ValueError("No valid molecules found in S-data")
    
    # Load tokenizer and model
    print("\n🔧 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_FILE, map_location=device)
    
    # Create model
    model = ChemBERTaFineTuned(
        model_name="seyonec/ChemBERTa-zinc-base-v1",
        hidden_size=checkpoint['config']['hidden_size']
    ).to(device)
    
    # Handle torch.compile() artifacts by stripping '_orig_mod.' prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("⚠️  Detected compiled model (_orig_mod prefix), stripping prefixes...")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('_orig_mod.'):
                new_key = k[10:]  # Remove '_orig_mod.' prefix
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # Also handle potential missing keys due to strict loading
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✅ Model loaded with strict matching")
    except RuntimeError as e:
        print(f"⚠️  Strict loading failed: {e}")
        print("Attempting non-strict loading...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  Missing keys: {missing_keys[:5]}...")  # Show first 5
        if unexpected_keys:
            print(f"  Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
        print("✅ Model loaded with non-strict matching")
    
    model.eval()
    
    print(f"✅ Model loaded (best val AUC on B3DB: {checkpoint['best_val_auc']:.4f})")
    
    # Create dataset and loader
    dataset = ChemBERTaDataset(smiles, labels, tokenizer)
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
    result_file = os.path.join(OUTPUT_DIR, f'chemberta_sdata_results_{timestamp}.json')
    preds_file = os.path.join(OUTPUT_DIR, f'chemberta_sdata_predictions_{timestamp}.csv')
    
    with open(result_file, 'w') as f:
        json.dump({
            'dataset': 'S-data',
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