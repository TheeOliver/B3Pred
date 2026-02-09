#!/usr/bin/env python3
"""Evaluation script for trained BBB predictor models."""
import argparse
import json
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader

from configs.base_config import BaseConfig
from src.data.featurizer import MoleculeDataset
from src.models.predictor import Predictor
from src.utils.evaluate import evaluate_model, print_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained BBB predictor models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on test set
  python scripts/evaluate.py --experiment gat_baseline --split test
  
  # Evaluate on validation set
  python scripts/evaluate.py --experiment gcn_exp1 --split val
  
  # Evaluate on all splits
  python scripts/evaluate.py --experiment sage_baseline --split all
        """
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        required=True,
        help='Name of experiment to evaluate'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test', 'all'],
        help='Dataset split to evaluate on'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for evaluation'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    exp_dir = BaseConfig.get_experiment_dir(args.experiment)
    config_path = BaseConfig.get_config_path(args.experiment)
    model_path = BaseConfig.get_model_path(args.experiment)
    
    # Check if experiment exists
    if not exp_dir.exists():
        print(f"Error: Experiment '{args.experiment}' not found")
        print(f"Expected directory: {exp_dir}")
        return
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    device = torch.device(args.device)
    
    print("\n" + "="*60)
    print(f"Evaluating {config['model_name']} - Experiment: {args.experiment}")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Device: {device}\n")
    
    # Determine which splits to evaluate
    splits_to_eval = []
    if args.split == 'all':
        splits_to_eval = ['train', 'val', 'test']
    else:
        splits_to_eval = [args.split]
    
    # Load model
    print("Loading model...")
    
    # Load a sample to get node dimension
    sample_df = pd.read_csv(BaseConfig.TRAIN_DATA).head(1)
    sample_dataset = MoleculeDataset(sample_df)
    node_dim = sample_dataset[0].x.shape[1]
    
    # Create and load model
    model = Predictor.from_config(config, node_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully\n")
    
    # Evaluate on each split
    all_results = {}
    
    for split in splits_to_eval:
        # Load data
        if split == 'train':
            data_path = BaseConfig.TRAIN_DATA
        elif split == 'val':
            data_path = BaseConfig.VAL_DATA
        else:  # test
            data_path = BaseConfig.TEST_DATA
        
        print(f"Loading {split} data from {data_path}...")
        df = pd.read_csv(data_path)
        dataset = MoleculeDataset(df)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        print(f"{split.capitalize()} samples: {len(dataset)}")
        
        # Evaluate
        metrics, preds, labels = evaluate_model(model, loader, device)
        print_metrics(metrics, f"{split.capitalize()}")
        
        all_results[split] = metrics
    
    # Save results
    results_path = exp_dir / f"evaluation_{args.split}.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
