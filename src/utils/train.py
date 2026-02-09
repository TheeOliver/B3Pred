"""Training utilities for BBB predictor."""
import torch
import numpy as np
from typing import Dict, Optional
from pathlib import Path
from src.utils.evaluate import evaluate_model, print_metrics
from configs.model_configs import TrainingConfig


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict,
    device: torch.device = torch.device('cpu'),
    save_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict:
    """
    Train a model with early stopping.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        config: Training configuration dictionary
        device: Device to train on
        save_path: Path to save best model (optional)
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    
    # Setup training
    criterion = TrainingConfig.LOSS_FUNCTIONS[config['loss']]()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Early stopping setup
    best_val_f1 = 0.0
    patience_counter = 0
    patience = config.get('early_stopping_patience', 10)
    
    # Training history
    history = {
        'train_loss': [],
        'val_f1': [],
        'val_accuracy': [],
        'val_auc': [],
    }
    
    if verbose:
        print(f"\nStarting training for {config['epochs']} epochs...")
        print(f"Device: {device}")
        print(f"Learning rate: {config['learning_rate']}")
        print(f"Batch size: {config['batch_size']}\n")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, data.y)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        
        history['val_f1'].append(val_metrics['f1'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc'] if val_metrics['auc'] else 0.0)
        
        if verbose:
            print(f"Epoch {epoch+1}/{config['epochs']} - "
                  f"Train Loss: {avg_train_loss:.4f} - "
                  f"Val F1: {val_metrics['f1']:.4f} - "
                  f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping and model saving
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            
            if save_path:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"  → Saved best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    if verbose:
        print(f"\nTraining complete. Best validation F1: {best_val_f1:.4f}")
    
    # Load best model if save_path was provided
    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path))
        if verbose:
            print(f"Loaded best model from {save_path}")
    
    return history
