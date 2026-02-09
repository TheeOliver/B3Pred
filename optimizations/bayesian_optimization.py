"""
Bayesian Optimization for BBB Prediction Models using Optuna
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE
"""

import json
import argparse
import sys
from pathlib import Path

# Add project root to Python path
import os
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch_geometric.loader import DataLoader

# Import from existing modules
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
from configs.predictor_config import GraphConfig


class BayesianOptimizer:
    """Bayesian Optimization using Optuna for GNN hyperparameter tuning"""

    def __init__(self, model_name: str, n_trials: int = 100, study_name: str = None,
                 results_dir: Path = None, seed: int = 42):
        """
        Initialize Bayesian Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_trials: Number of optimization trials
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.study_name = study_name or f"{model_name}_bayesian_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "bayesian_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load datasets once
        print("Loading datasets...")
        self.train_df = pd.read_csv(settings.TRAIN_DATA)
        self.val_df = pd.read_csv(settings.VAL_DATA)
        self.test_df = pd.read_csv(settings.TEST_DATA)

        # Prepare datasets
        self.train_dataset = MoleculeDataset(data=self.train_df)
        self.stats = compute_feature_stats(self.train_dataset)
        normalize_dataset(self.train_dataset, self.stats)

        self.val_dataset = MoleculeDataset(data=self.val_df)
        normalize_dataset(self.val_dataset, self.stats)

        self.test_dataset = MoleculeDataset(data=self.test_df)
        normalize_dataset(self.test_dataset, self.stats)

        # Get graph info including edge_dim
        sample = self.train_dataset[0]
        self.graph_info = {
            'node_dim': sample.x.shape[1],
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 4
        }

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")

        # Results tracking
        self.all_results = []
        self.best_config = None
        self.best_score = -float('inf')

    def suggest_hyperparameters(self, trial: optuna.Trial) -> dict:
        """
        Suggest hyperparameters based on model type

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        config = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_trial_{trial.number}",
        }

        # Common hyperparameters for all models
        config['graph_layers'] = trial.suggest_int('graph_layers', 2, 5)
        config['graph_hidden_channels'] = trial.suggest_categorical(
            'graph_hidden_channels', [32, 64, 128, 256, 512]
        )
        config['graph_dropouts'] = trial.suggest_float('graph_dropouts', 0.0, 0.6)
        config['graph_norm'] = trial.suggest_categorical('graph_norm', [True, False])

        # Predictor hyperparameters
        config['pred_layers'] = trial.suggest_int('pred_layers', 2, 4)
        config['pred_hidden_channels'] = trial.suggest_categorical(
            'pred_hidden_channels', [32, 64, 128, 256]
        )
        config['pred_dropouts'] = trial.suggest_float('pred_dropouts', 0.0, 0.6)

        # Training hyperparameters
        config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        config['lr'] = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        config['epochs'] = trial.suggest_int('epochs', 20, 100)
        config['loss'] = 'crossentropy'
        config['subset_size'] = 1.0

        # Model-specific hyperparameters
        if self.model_name == 'GAT':
            config['attention_heads'] = trial.suggest_categorical('attention_heads', [2, 4, 8])
            config['attention_dropouts'] = trial.suggest_float('attention_dropouts', 0.0, 0.6)

        return config

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization

        Args:
            trial: Optuna trial object

        Returns:
            Validation F1 score (to maximize)
        """
        try:
            # Get hyperparameters
            config = self.suggest_hyperparameters(trial)

            # Create data loaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config['batch_size'],
                shuffle=False
            )

            # Build model
            model_class = GraphConfig.models[self.model_name]['model']
            model = model_class.from_config(config, self.graph_info)

            # Train model
            res, trained_model = train_model(
                model,
                train_loader,
                val_loader,
                config['epochs'],
                [settings.TARGET_LABEL],
                loss_type=config['loss'],
                learning_rate=config['lr'],
                hetero=False,
                log=False,
                save_to=None
            )

            # Extract validation F1 score
            val_f1 = res['macro_f1']

            # Store results
            result_entry = {
                'trial': trial.number,
                'val_f1': float(val_f1),
                'config': config,
                'all_metrics': res
            }
            self.all_results.append(result_entry)

            # Update best config
            if val_f1 > self.best_score:
                self.best_score = val_f1
                self.best_config = config
                print(f"\n{'='*70}")
                print(f"NEW BEST SCORE: {val_f1:.4f}")
                print(f"{'='*70}\n")

            return val_f1

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return -1.0

    def run_optimization(self):
        """Run Bayesian optimization"""
        print(f"\n{'='*70}")
        print(f"STARTING BAYESIAN OPTIMIZATION FOR {self.model_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"{'='*70}\n")

        # Create Optuna study
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=TPESampler(seed=self.seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )

        # Run optimization
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        # Get best trial
        best_trial = study.best_trial

        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best trial: {best_trial.number}")
        print(f"Best validation F1: {best_trial.value:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key:30s}: {value}")
        print(f"{'='*70}\n")

        return study

    def evaluate_best_model(self):
        """Evaluate best model on test set"""
        if self.best_config is None:
            print("No best config found. Run optimization first.")
            return None

        print(f"\n{'='*70}")
        print(f"EVALUATING BEST MODEL ON TEST SET")
        print(f"{'='*70}\n")

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.best_config['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.best_config['batch_size'],
            shuffle=False
        )

        # Build and train model with best config
        model_class = GraphConfig.models[self.model_name]['model']
        model = model_class.from_config(self.best_config, self.graph_info)

        # Train on full training set
        _, trained_model = train_model(
            model,
            train_loader,
            test_loader,
            self.best_config['epochs'],
            [settings.TARGET_LABEL],
            loss_type=self.best_config['loss'],
            learning_rate=self.best_config['lr'],
            hetero=False,
            log=False,
            save_to=None
        )

        # Evaluate on test set
        test_results = test_model(
            test_loader,
            trained_model,
            [settings.TEST_LABEL]
        )

        print(f"\n{'='*70}")
        print(f"TEST SET RESULTS")
        print(f"{'='*70}")
        for key, value in test_results.items():
            if value is not None:
                print(f"{key:30s}: {value:.4f}")
        print(f"{'='*70}\n")

        return test_results

    def save_results(self, study: optuna.Study, test_results: dict = None):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'n_trials': self.n_trials,
                'best_score': float(self.best_score),
                'best_config': self.best_config,
                'test_results': test_results,
                'all_trials': self.all_results
            }, f, indent=2)

        print(f"Detailed results saved to: {results_file}")

        # Save summary to text file with highlighted best results
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BAYESIAN OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Total Trials: {self.n_trials}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            f.write("="*80 + "\n")
            f.write("*** BEST CONFIGURATION (VALIDATION) ***\n")
            f.write("="*80 + "\n")
            f.write(f"Best Validation F1 Score: {self.best_score:.4f}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in self.best_config.items():
                f.write(f"  {key:30s}: {value}\n")
            f.write("\n")

            if test_results:
                f.write("="*80 + "\n")
                f.write("*** TEST SET PERFORMANCE ***\n")
                f.write("="*80 + "\n")
                for key, value in test_results.items():
                    if value is not None:
                        f.write(f"  {key:30s}: {value:.4f}\n")
                f.write("\n")

            # Top 10 trials
            sorted_results = sorted(self.all_results, key=lambda x: x['val_f1'], reverse=True)
            f.write("="*80 + "\n")
            f.write("TOP 10 TRIALS\n")
            f.write("="*80 + "\n")
            for i, result in enumerate(sorted_results[:10], 1):
                f.write(f"\nRank {i}: Trial {result['trial']}\n")
                f.write(f"  Validation F1: {result['val_f1']:.4f}\n")
                f.write(f"  Key hyperparameters:\n")
                config = result['config']
                f.write(f"    graph_layers: {config.get('graph_layers')}\n")
                f.write(f"    graph_hidden_channels: {config.get('graph_hidden_channels')}\n")
                f.write(f"    learning_rate: {config.get('lr'):.6f}\n")
                f.write(f"    batch_size: {config.get('batch_size')}\n")
                if 'attention_heads' in config:
                    f.write(f"    attention_heads: {config.get('attention_heads')}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("="*80 + "\n")

        print(f"Summary saved to: {summary_file}")

        # Save Optuna study
        study_file = self.results_dir / f"{self.study_name}_{timestamp}_study.pkl"
        import pickle
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)

        print(f"Optuna study saved to: {study_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the optimization study')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--evaluate_test', action='store_true',
                        help='Evaluate best model on test set after optimization')

    args = parser.parse_args()

    # Create optimizer
    optimizer = BayesianOptimizer(
        model_name=args.model,
        n_trials=args.n_trials,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed
    )

    # Run optimization
    study = optimizer.run_optimization()

    # Evaluate on test set if requested
    test_results = None
    if args.evaluate_test:
        test_results = optimizer.evaluate_best_model()

    # Save results
    optimizer.save_results(study, test_results)

    print("\n" + "="*80)
    print("BAYESIAN OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()