"""
ChemBERTa Fine-tuning for b3db_tanimoto
========================================
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss, average_precision_score,
)

warnings.filterwarnings('ignore')

# ============== CONFIGURATION ==============

DATA_PATH  = os.environ.get('CHEMBERTA_DATA_PATH', '/manu/data/b3db_tanimoto')
OUTPUT_DIR = os.environ.get('CHEMBERTA_OUTPUT_DIR', '/manu/results/chemberta_mydata')
DATASET_NAME = os.environ.get('CHEMBERTA_DATASET_NAME', 'b3db_tanimoto')
TASK_TYPE    = 'classification'

SEED            = int(os.environ.get('CHEMBERTA_SEED', 42))
MAX_EPOCHS      = 50
PATIENCE        = 15
BATCH_SIZE      = 256   # A100-80GB: ChemBERTa (~45M params) fits large batches easily
NUM_WORKERS     = 4     # Parallel CPU workers to keep the GPU fed
UNFREEZE_LAYERS = 2     # Unfreeze last 2 transformer layers
LR_ENCODER      = 1e-5  # Lower LR for pretrained layers
LR_HEAD         = 1e-3  # Normal LR for prediction head


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

    def extract(df):
        data = [(s, float(l)) for s, l in zip(df['Drug'], df['Y']) if not np.isnan(float(l))]
        smiles, labels = zip(*data) if data else ([], [])
        return list(smiles), list(labels)

    train_smiles, train_labels = extract(train_df)
    val_smiles,   val_labels   = extract(val_df)
    test_smiles,  test_labels  = extract(test_df)

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


class ChemBERTaDataset(Dataset):
    def __init__(self, smiles_list, labels, tokenizer, max_length=128):
        self.smiles     = smiles_list
        self.labels     = labels
        self.tokenizer  = tokenizer
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
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.float32)
        }


class ChemBERTaFineTuned(nn.Module):
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base-v1",
                 unfreeze_layers=2, dropout=0.2):
        super().__init__()

        print(f"  Loading ChemBERTa from {model_name}...")
        self.encoder     = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        for param in self.encoder.parameters():
            param.requires_grad = False

        total_layers = len(self.encoder.encoder.layer)
        for i in range(total_layers - unfreeze_layers, total_layers):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = True

        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for param in self.encoder.pooler.parameters():
                param.requires_grad = True

        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled  = outputs.last_hidden_state[:, 0, :]
        return self.head(pooled)


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0
    n_samples  = 0

    for batch in loader:
        input_ids      = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels         = batch['label'].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask)
            loss    = criterion(outputs.squeeze(), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

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


def evaluate_classification(model, loader, device, threshold=0.5, return_preds=False):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels         = batch['label'].to(device, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs.squeeze().float())

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


def run_finetune_for_my_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*70}")
    print("CHEMBERTA FINE-TUNING — b3db_tanimoto")
    print(f"{'='*70}")
    print(f"Device         : {device}")
    print(f"Batch size     : {BATCH_SIZE}")
    print(f"Unfreeze layers: {UNFREEZE_LAYERS}")
    print(f"LR encoder/head: {LR_ENCODER} / {LR_HEAD}")
    print(f"Max epochs     : {MAX_EPOCHS}  Patience: {PATIENCE}")
    print(f"Output dir     : {OUTPUT_DIR}")
    print(f"{'='*70}\n")

    set_seed(SEED)
    data = load_my_data()

    tokenizer     = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    train_dataset = ChemBERTaDataset(data['train_smiles'], data['train_labels'], tokenizer)
    val_dataset   = ChemBERTaDataset(data['val_smiles'],   data['val_labels'],   tokenizer)
    test_dataset  = ChemBERTaDataset(data['test_smiles'],  data['test_labels'],  tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE * 2,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE * 2,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    model = ChemBERTaFineTuned(unfreeze_layers=UNFREEZE_LAYERS, dropout=0.2).to(device)
    if hasattr(torch, 'compile'):
        print("  Compiling model with torch.compile...")
        model = torch.compile(model)

    encoder_params = [p for n, p in model.named_parameters() if 'encoder' in n and p.requires_grad]
    head_params    = [p for n, p in model.named_parameters() if 'head' in n]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': LR_ENCODER, 'weight_decay': 0.01},
        {'params': head_params,    'lr': LR_HEAD,    'weight_decay': 0.01}
    ])

    total_steps = len(train_loader) * MAX_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    n_pos = sum(1 for y in data['train_labels'] if y > 0.5)
    n_neg = len(data['train_labels']) - n_pos
    pos_weight = torch.tensor([n_neg / n_pos], device=device) if n_pos > 0 and n_neg > 0 \
                 else torch.tensor([1.0], device=device)
    print(f"\n📊 Class weighting: pos_weight = {pos_weight.item():.2f}")

    amp_scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc     = 0
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'val_metrics': []}

    print("\n🚀 Starting training...")
    print("-" * 70)

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss  = train_epoch(model, train_loader, optimizer, criterion, device, scheduler, amp_scaler)
        val_metrics = evaluate_classification(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_metrics'].append({
            'epoch':    epoch,
            'auc_roc':  val_metrics['auc_roc'],
            'auc_pr':   val_metrics['auc_pr'],
            'f1_score': val_metrics['f1_score'],
            'accuracy': val_metrics['accuracy'],
            'log_loss': val_metrics['log_loss'],
        })

        print(f"Epoch {epoch:2d}: loss={train_loss:.4f} | "
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
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(OUTPUT_DIR, f'chemberta_model_{timestamp}.pt')
        
        torch.save({
            'model_state_dict': best_model_state,
            'config': {
                'unfreeze_layers': UNFREEZE_LAYERS,
                'hidden_size': model.hidden_size,
            },
            'best_val_auc': best_val_auc,
        }, model_path)
        print(f"💾 Model saved to: {model_path}")
        
        # Save tokenizer separately
        tokenizer_path = os.path.join(OUTPUT_DIR, f'tokenizer_{timestamp}')
        tokenizer.save_pretrained(tokenizer_path)
        print(f"💾 Tokenizer saved to: {tokenizer_path}")
        # ========== END ADDED SECTION ==========

    print("\n" + "=" * 70)
    print("📊 FINAL TEST SET EVALUATION")
    print("=" * 70)

    test_metrics = evaluate_classification(model, test_loader, device, return_preds=True)

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
    result_file = os.path.join(OUTPUT_DIR, f'chemberta_results_{timestamp}.json')
    preds_file  = os.path.join(OUTPUT_DIR, f'chemberta_predictions_{timestamp}.csv')

    with open(result_file, 'w') as f:
        json.dump({
            'dataset':        DATASET_NAME,
            'task_type':      TASK_TYPE,
            'train_size':     len(data['train_smiles']),
            'val_size':       len(data['val_smiles']),
            'test_size':      len(data['test_smiles']),
            'best_val_auc':   best_val_auc,
            'epochs_trained': len(history['train_loss']),
            'config': {
                'batch_size':      BATCH_SIZE,
                'unfreeze_layers': UNFREEZE_LAYERS,
                'lr_encoder':      LR_ENCODER,
                'lr_head':         LR_HEAD,
                'max_epochs':      MAX_EPOCHS,
                'patience':        PATIENCE,
                'seed':            SEED,
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
        run_finetune_for_my_data()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)