#!/usr/bin/env python3
"""Main training script for BBB predictor models."""
import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

from configs.base_config import BaseConfig
from configs.model_configs import (
    MODEL_CONFIGS,
    PredictorConfig,
    TrainingConfig
)
from src.data.featurizer import MoleculeDataset
from src.models.predictor import Predictor
from src.utils.train import train_model
from src.utils.evaluate import evaluate_model, print_metrics


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: dict, save_path: Path):
    """Save configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def merge_configs(file_config: dict, cli_args: dict) -> dict:
    """Merge file config with command-line arguments."""
    config = file_config.copy()
    
    # Override with CLI args if provided
    for key, value in cli_args.items():
        if value is not None:
            config[key] = value
    
    return config


def get_default_config(model_name: str) -> dict:
    """Get default configuration for a model."""
    config = {}
    
    # Add model-specific defaults
    if model_name in MODEL_CONFIGS:
        config.update(MODEL_CONFIGS[model_name].default_params)
    
    # Add predictor defaults
    config.update(PredictorConfig.default_params)
    
    # Add training defaults
    config.update(TrainingConfig.default_params)
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train BBB permeability prediction models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GAT model with default settings
  python scripts/train.py --model GAT --name gat_baseline
  
  # Train with custom config file
  python scripts/train.py --config configs/experiment_configs/my_config.json
  
  # Train with custom hyperparameters
  python scripts/train.py --model GCN --name gcn_exp1 --epochs 100 --lr 0.001
  
  # Train and evaluate on test set
  python scripts/train.py --model GraphSAGE --name sage_baseline --test
        """
    )
    
    # Config file or model specification
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file'
    )
    config_group.add_argument(
        '--model',
        type=str,
        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN'],
        help='Model architecture to train'
    )
    
    # Experiment settings
    parser.add_argument(
        '--name',
        type=str,
        help='Experiment name (required if using --model)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Evaluate on test set after training'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--lr', '--learning_rate', type=float, dest='learning_rate', 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, help='Weight decay')
    
    # Model hyperparameters
    parser.add_argument('--graph_layers', type=int, help='Number of graph layers')
    parser.add_argument('--graph_hidden_channels', type=int, 
                       help='Hidden channels in graph layers')
    parser.add_argument('--graph_dropout', type=float, help='Dropout in graph layers')
    parser.add_argument('--pred_layers', type=int, help='Number of prediction layers')
    parser.add_argument('--pred_hidden_channels', type=int,
                       help='Hidden channels in prediction layers')
    parser.add_argument('--pred_dropout', type=float, help='Dropout in prediction layers')
    
    # GAT-specific
    parser.add_argument('--attention_heads', type=int, help='Number of attention heads (GAT)')
    parser.add_argument('--attention_dropout', type=float, 
                       help='Attention dropout (GAT)')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config(Path(args.config))
        experiment_name = config.get('experiment_name', 'experiment')
    else:
        if not args.name:
            parser.error("--name is required when using --model")
        
        experiment_name = args.name
        config = get_default_config(args.model)
        config['experiment_name'] = experiment_name
        
        # Merge CLI arguments
        cli_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'graph_layers': args.graph_layers,
            'graph_hidden_channels': args.graph_hidden_channels,
            'graph_dropout': args.graph_dropout,
            'pred_layers': args.pred_layers,
            'pred_hidden_channels': args.pred_hidden_channels,
            'pred_dropout': args.pred_dropout,
            'attention_heads': args.attention_heads,
            'attention_dropout': args.attention_dropout,
        }
        config = merge_configs(config, cli_config)
    
    # Setup device
    device = torch.device(args.device)
    
    print("\n" + "="*60)
    print(f"Training {config['model_name']} - Experiment: {experiment_name}")
    print("="*60)
    print(f"\nConfiguration:")
    for key, value in sorted(config.items()):
        print(f"  {key}: {value}")
    print()
    
    # Create experiment directory and save config
    exp_dir = BaseConfig.get_experiment_dir(experiment_name)
    save_config(config, BaseConfig.get_config_path(experiment_name))
    print(f"Experiment directory: {exp_dir}")
    print(f"Config saved to: {BaseConfig.get_config_path(experiment_name)}\n")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(BaseConfig.TRAIN_DATA)
    val_df = pd.read_csv(BaseConfig.VAL_DATA)
    
    train_dataset = MoleculeDataset(train_df)
    val_dataset = MoleculeDataset(val_df)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get node dimension from first sample
    node_dim = train_dataset[0].x.shape[1]
    print(f"Node feature dimension: {node_dim}\n")
    
    # Create model
    print(f"Creating {config['model_name']} model...")
    model = Predictor.from_config(config, node_dim)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Train model
    model_save_path = BaseConfig.get_model_path(experiment_name)
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        save_path=model_save_path,
        verbose=True
    )
    
    # Final validation evaluation
    print("\nFinal validation evaluation:")
    val_metrics, _, _ = evaluate_model(model, val_loader, device)
    print_metrics(val_metrics, "Validation")
    
    # Save validation results
    results = {
        'validation': val_metrics,
        'history': history
    }
    
    # Test evaluation if requested
    if args.test:
        print("\nEvaluating on test set...")
        test_df = pd.read_csv(BaseConfig.TEST_DATA)
        test_dataset = MoleculeDataset(test_df)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False
        )
        
        test_metrics, _, _ = evaluate_model(model, test_loader, device)
        print_metrics(test_metrics, "Test")
        results['test'] = test_metrics
    
    # Save all results
    results_path = BaseConfig.get_results_path(experiment_name)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Model saved to: {model_save_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
