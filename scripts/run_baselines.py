#!/usr/bin/env python3
"""Run traditional ML baseline models for BBB prediction."""
import argparse
import json
import pandas as pd
import pickle
from pathlib import Path

from configs.base_config import BaseConfig
from src.baselines import (
    build_feature_matrix,
    evaluate_baseline,
    get_logistic_regression,
    get_random_forest,
    get_svm,
)


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate baseline ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all baseline models
  python scripts/run_baselines.py --all
  
  # Train specific model
  python scripts/run_baselines.py --model logistic
  
  # Train with custom fingerprint size
  python scripts/run_baselines.py --all --n_bits 1024
        """
    )
    
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--all',
        action='store_true',
        help='Train all baseline models'
    )
    model_group.add_argument(
        '--model',
        type=str,
        choices=['logistic', 'random_forest', 'svm'],
        help='Specific model to train'
    )
    
    parser.add_argument(
        '--n_bits',
        type=int,
        default=2048,
        help='Number of bits for Morgan fingerprint (default: 2048)'
    )
    
    args = parser.parse_args()
    
    # Determine which models to train
    if args.all:
        models_to_train = ['logistic', 'random_forest', 'svm']
    else:
        models_to_train = [args.model]
    
    print("\n" + "="*60)
    print("BBB Baseline Models Training")
    print("="*60)
    print(f"Models: {', '.join(models_to_train)}")
    print(f"Fingerprint bits: {args.n_bits}\n")
    
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv(BaseConfig.TRAIN_DATA)
    val_df = pd.read_csv(BaseConfig.VAL_DATA)
    test_df = pd.read_csv(BaseConfig.TEST_DATA)
    
    # Build feature matrices
    print("Building feature matrices...")
    X_train, y_train = build_feature_matrix(train_df, n_bits=args.n_bits)
    X_val, y_val = build_feature_matrix(val_df, n_bits=args.n_bits)
    X_test, y_test = build_feature_matrix(test_df, n_bits=args.n_bits)
    
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}\n")
    
    # Model configurations
    model_configs = {
        'logistic': {
            'name': 'Logistic Regression',
            'getter': get_logistic_regression,
        },
        'random_forest': {
            'name': 'Random Forest',
            'getter': get_random_forest,
        },
        'svm': {
            'name': 'Support Vector Machine',
            'getter': get_svm,
        }
    }
    
    # Train and evaluate each model
    all_results = {}
    
    for model_key in models_to_train:
        config = model_configs[model_key]
        
        print("="*60)
        print(f"Training {config['name']}")
        print("="*60)
        
        # Create model
        model = config['getter']()
        
        # Train
        print("Training...")
        model.fit(X_train, y_train)
        print("Training complete!\n")
        
        # Evaluate on all splits
        train_metrics = evaluate_baseline(model, X_train, y_train, "Train")
        val_metrics = evaluate_baseline(model, X_val, y_val, "Validation")
        test_metrics = evaluate_baseline(model, X_test, y_test, "Test")
        
        # Save results
        results = {
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics,
            'config': {
                'model': config['name'],
                'n_bits': args.n_bits,
            }
        }
        
        all_results[model_key] = results
        
        # Save model
        exp_dir = BaseConfig.get_experiment_dir(f"baseline_{model_key}")
        model_path = exp_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to: {model_path}")
        
        # Save results
        results_path = exp_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}\n")
    
    # Save combined results
    combined_path = BaseConfig.EXPERIMENTS_DIR / "baseline_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("="*60)
    print(f"All results saved to: {combined_path}")
    print("Baseline training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
