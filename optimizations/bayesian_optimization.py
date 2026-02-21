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
                 results_dir: Path = None, seed: int = 42,
                 opt_subset_size: float = 0.1, opt_epochs: int = 10,
                 top_k: int = 10, full_epochs: int = 100):
        """
        Initialize Bayesian Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_trials: Number of optimization trials
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
            opt_subset_size: Fraction of training data to use during optimization
            opt_epochs: Number of epochs to use during optimization
            top_k: Number of top configurations to retrain
            full_epochs: Number of epochs for full training
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.study_name = study_name or f"{model_name}_bayesian_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "bayesian_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Optimization strategy parameters
        self.opt_subset_size = opt_subset_size
        self.opt_epochs = opt_epochs
        self.top_k = top_k
        self.full_epochs = full_epochs

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

        # Create fixed random subset for optimization
        if self.opt_subset_size < 1.0:
            n_subset = int(len(self.train_dataset) * self.opt_subset_size)
            # Use fixed seed for reproducibility
            generator = torch.Generator().manual_seed(self.seed)
            subset_indices = torch.randperm(len(self.train_dataset), generator=generator)[:n_subset]
            self.train_subset = torch.utils.data.Subset(self.train_dataset, subset_indices)
            print(f"Using {len(self.train_subset)}/{len(self.train_dataset)} samples for optimization ({self.opt_subset_size*100:.1f}%)")
        else:
            self.train_subset = self.train_dataset
            print("Using full training dataset for optimization")

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
        
        # Set fixed optimization epochs and subset size
        config['epochs'] = self.opt_epochs
        config['subset_size'] = self.opt_subset_size
        
        config['loss'] = 'crossentropy'

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
                self.train_subset,  # Use subset for optimization
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
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
        print(f"Using {self.opt_subset_size*100:.1f}% of data for {self.opt_epochs} epochs per trial")
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

    def retrain_top_k(self, k=None, full_epochs=None):
        """
        Retrain top-k configurations on full dataset and pick the best.
        
        Args:
            k: Number of top configurations to retrain (default: self.top_k)
            full_epochs: Number of epochs for full training (default: self.full_epochs)
        
        Returns:
            Test results for the best model
        """
        if k is None:
            k = self.top_k
        if full_epochs is None:
            full_epochs = self.full_epochs
            
        if not hasattr(self, 'all_results') or not self.all_results:
            print("No results available to retrain. Run optimization first.")
            return None

        # Sort by validation F1 (descending)
        sorted_results = sorted(self.all_results, key=lambda x: x['val_f1'], reverse=True)
        top_configs = [r['config'] for r in sorted_results[:k]]

        print(f"\n{'='*70}")
        print(f"RETRAINING TOP {k} CONFIGURATIONS ON FULL DATASET")
        print(f"Full training epochs: {full_epochs}")
        print(f"{'='*70}\n")
        
        best_val_f1 = -float('inf')
        best_config = None
        best_model = None
        best_train_results = None
        retrain_results = []

        for i, config in enumerate(top_configs):
            print(f"\n[{i+1}/{k}] Retraining configuration...")
            
            # Override epochs to use full training epochs
            config['epochs'] = full_epochs
            config['subset_size'] = 1.0  # Use full data
            
            # Create data loaders with full dataset
            train_loader = DataLoader(
                self.train_dataset,  # Use full training set
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=torch.cuda.is_available()
            )

            # Build model
            model_class = GraphConfig.models[self.model_name]['model']
            model = model_class.from_config(config, self.graph_info)

            # Train on full dataset
            print(f"  Training for {full_epochs} epochs...")
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
            
            val_f1 = res['macro_f1']
            print(f"  Validation F1 after full training: {val_f1:.4f}")
            
            # Store retraining result
            retrain_result = {
                'original_trial': config.get('config_name', f'trial_{i}'),
                'config': config,
                'val_f1': float(val_f1),
                'all_metrics': res
            }
            retrain_results.append(retrain_result)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = config
                best_model = trained_model
                best_train_results = res
                print(f"  → New best among top-{k}!")

        # Evaluate best model on test set
        print(f"\n{'='*70}")
        print(f"Evaluating best model on test set")
        print(f"{'='*70}")
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=best_config['batch_size'],
            shuffle=False
        )
        
        test_results = test_model(
            test_loader,
            best_model,
            [settings.TEST_LABEL]
        )

        print(f"\nBest validation F1 after full training: {best_val_f1:.4f}")
        print(f"Test results:")
        for key, value in test_results.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")

        # Update global best with the retrained model
        self.global_best_config = best_config
        self.global_best_fitness = best_val_f1
        
        # Store retraining results
        self.retrain_results = retrain_results
        self.final_test_results = test_results

        return test_results

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

        # Prepare data dictionary
        data = {
            'model_name': self.model_name,
            'n_trials': self.n_trials,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'best_score': float(self.best_score),
            'best_config': self.best_config,
            'test_results': test_results,
            'all_trials': self.all_results
        }
        
        # Add retraining results if available
        if hasattr(self, 'retrain_results'):
            data['retrain_results'] = self.retrain_results
        if hasattr(self, 'final_test_results'):
            data['final_test_results'] = self.final_test_results

        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Detailed results saved to: {results_file}")

        # Save summary to text file with highlighted best results
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"BAYESIAN OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Total Trials: {self.n_trials}\n")
            f.write(f"Optimization Data Subset: {self.opt_subset_size*100:.1f}%\n")
            f.write(f"Optimization Epochs: {self.opt_epochs}\n")
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

            # Add retraining section if available
            if hasattr(self, 'retrain_results'):
                f.write("\n" + "="*80 + "\n")
                f.write(f"*** RETRAINING RESULTS (TOP-{self.top_k} ON FULL DATA) ***\n")
                f.write("="*80 + "\n\n")
                
                sorted_retrain = sorted(self.retrain_results, key=lambda x: x['val_f1'], reverse=True)
                for i, res in enumerate(sorted_retrain, 1):
                    f.write(f"Rank {i}:\n")
                    f.write(f"  Validation F1: {res['val_f1']:.4f}\n")
                    if i == 1 and hasattr(self, 'final_test_results'):
                        f.write(f"  Test F1: {self.final_test_results.get('macro_f1', 0):.4f}\n")
                        f.write(f"  Test Accuracy: {self.final_test_results.get('acc_' + self.model_name, 0):.4f}\n")
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
    
    # New arguments for optimization strategy
    parser.add_argument('--subset_size', type=float, default=0.1,
                        help='Fraction of training data to use during optimization (default: 0.1)')
    parser.add_argument('--opt_epochs', type=int, default=10,
                        help='Number of epochs to use during optimization (default: 10)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top configurations to retrain on full data (default: 10)')
    parser.add_argument('--full_epochs', type=int, default=100,
                        help='Number of epochs for full training (default: 100)')
    parser.add_argument('--auto_retrain', action='store_true',
                        help='Automatically retrain top-k models after optimization')

    args = parser.parse_args()

    # Create optimizer
    optimizer = BayesianOptimizer(
        model_name=args.model,
        n_trials=args.n_trials,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed,
        opt_subset_size=args.subset_size,
        opt_epochs=args.opt_epochs,
        top_k=args.top_k,
        full_epochs=args.full_epochs
    )

    # Run optimization
    study = optimizer.run_optimization()

    # Auto-retrain if requested
    if args.auto_retrain:
        print("\n" + "="*70)
        print("AUTOMATICALLY RETRAINING TOP MODELS ON FULL DATA")
        print("="*70)
        final_test_results = optimizer.retrain_top_k(
            k=args.top_k,
            full_epochs=args.full_epochs
        )
        test_results = final_test_results
    else:
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