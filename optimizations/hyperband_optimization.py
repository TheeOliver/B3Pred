"""
Hyperband Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE
Uses successive halving for efficient hyperparameter search
"""

import json
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import RandomSampler
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing modules
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
from configs.predictor_config import GraphConfig


class HyperbandOptimizer:
    """Hyperband Optimization for efficient hyperparameter search"""

    def __init__(self, model_name: str, n_trials: int = 100, max_epochs: int = 81,
                 reduction_factor: int = 3, study_name: str = None,
                 results_dir: Path = None, seed: int = 42):
        """
        Initialize Hyperband Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_trials: Number of optimization trials
            max_epochs: Maximum epochs for training (should be divisible by reduction_factor^n)
            reduction_factor: Factor by which to reduce number of configurations (default: 3)
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.max_epochs = max_epochs
        self.reduction_factor = reduction_factor
        self.study_name = study_name or f"{model_name}_hyperband_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "hyperband_optimization"
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

        self.graph_info = {'node_dim': self.train_dataset[0].x.shape[1]}

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")

        # Results tracking
        self.all_results = []
        self.best_config = None
        self.best_score = -float('inf')

        # Hyperband brackets
        self.brackets_info = self._compute_hyperband_brackets()

    def _compute_hyperband_brackets(self):
        """Compute Hyperband bracket structure"""
        s_max = int(np.log(self.max_epochs) / np.log(self.reduction_factor))
        brackets = []

        for s in range(s_max, -1, -1):
            n = int(np.ceil((s_max + 1) / (s + 1) * self.reduction_factor ** s))
            r = self.max_epochs * self.reduction_factor ** (-s)
            brackets.append({
                'bracket': s,
                'n_configs': n,
                'initial_epochs': int(r),
                'max_epochs': self.max_epochs
            })

        return brackets

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
        config['epochs'] = self.max_epochs  # Will be managed by Hyperband
        config['loss'] = 'crossentropy'
        config['subset_size'] = 1.0

        # Model-specific hyperparameters
        if self.model_name == 'GAT':
            config['attention_heads'] = trial.suggest_categorical('attention_heads', [2, 4, 8])
            config['attention_dropouts'] = trial.suggest_float('attention_dropouts', 0.0, 0.6)

        return config

    def objective_with_pruning(self, trial: optuna.Trial) -> float:
        """
        Objective function with Hyperband pruning

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

            # Setup training
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=5e-4)

            best_val_f1 = -float('inf')

            # Train with early reporting for pruning
            for epoch in range(config['epochs']):
                # Training phase
                model.train()
                losses = []
                for data in train_loader:
                    optimizer.zero_grad()
                    out = model(data)
                    loss = criterion(out, data.y.view(-1))
                    losses.append(loss.item())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                # Validation phase
                res = test_model(val_loader, model, [settings.TARGET_LABEL], hetero=False)
                val_f1 = res['macro_f1']

                # Report intermediate value for pruning
                trial.report(val_f1, epoch)

                # Check if trial should be pruned
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

                # Track best validation score
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

            # Store results
            result_entry = {
                'trial': trial.number,
                'val_f1': float(best_val_f1),
                'final_epoch': epoch + 1,
                'config': config,
                'all_metrics': res
            }
            self.all_results.append(result_entry)

            # Update best config
            if best_val_f1 > self.best_score:
                self.best_score = best_val_f1
                self.best_config = config
                print(f"\n{'='*70}")
                print(f"NEW BEST SCORE: {best_val_f1:.4f} at Trial {trial.number}")
                print(f"{'='*70}\n")

            return best_val_f1

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return -1.0

    def run_optimization(self):
        """Run Hyperband optimization"""
        print(f"\n{'='*70}")
        print(f"STARTING HYPERBAND OPTIMIZATION FOR {self.model_name}")
        print(f"Number of trials: {self.n_trials}")
        print(f"Max epochs: {self.max_epochs}")
        print(f"Reduction factor: {self.reduction_factor}")
        print(f"{'='*70}\n")

        print("Hyperband Bracket Structure:")
        for bracket in self.brackets_info:
            print(f"  Bracket {bracket['bracket']}: {bracket['n_configs']} configs, "
                  f"{bracket['initial_epochs']} initial epochs")
        print()

        # Create Optuna study with Hyperband pruner
        study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=RandomSampler(seed=self.seed),
            pruner=HyperbandPruner(
                min_resource=1,
                max_resource=self.max_epochs,
                reduction_factor=self.reduction_factor
            )
        )

        # Run optimization
        study.optimize(self.objective_with_pruning, n_trials=self.n_trials,
                      show_progress_bar=True)

        # Get best trial
        best_trial = study.best_trial

        print(f"\n{'='*70}")
        print(f"HYPERBAND OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Best trial: {best_trial.number}")
        print(f"Best validation F1: {best_trial.value:.4f}")
        print(f"Total trials completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"Total trials pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
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

        # Calculate statistics
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'n_trials': self.n_trials,
                'max_epochs': self.max_epochs,
                'reduction_factor': self.reduction_factor,
                'best_score': float(self.best_score),
                'best_config': self.best_config,
                'test_results': test_results,
                'completed_trials': len(completed_trials),
                'pruned_trials': len(pruned_trials),
                'brackets_info': self.brackets_info,
                'all_trials': self.all_results
            }, f, indent=2)

        print(f"Detailed results saved to: {results_file}")

        # Save summary to text file with highlighted best results
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"HYPERBAND OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Total Trials: {self.n_trials}\n")
            f.write(f"Max Epochs: {self.max_epochs}\n")
            f.write(f"Reduction Factor: {self.reduction_factor}\n")
            f.write(f"Completed Trials: {len(completed_trials)}\n")
            f.write(f"Pruned Trials: {len(pruned_trials)}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            f.write("Hyperband Bracket Structure:\n")
            for bracket in self.brackets_info:
                f.write(f"  Bracket {bracket['bracket']}: {bracket['n_configs']} configs, "
                       f"{bracket['initial_epochs']} initial epochs → {bracket['max_epochs']} max epochs\n")
            f.write("\n")

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
                f.write(f"  Epochs Completed: {result.get('final_epoch', 'N/A')}\n")
                f.write(f"  Key hyperparameters:\n")
                config = result['config']
                f.write(f"    graph_layers: {config.get('graph_layers')}\n")
                f.write(f"    graph_hidden_channels: {config.get('graph_hidden_channels')}\n")
                f.write(f"    learning_rate: {config.get('lr'):.6f}\n")
                f.write(f"    batch_size: {config.get('batch_size')}\n")
                if 'attention_heads' in config:
                    f.write(f"    attention_heads: {config.get('attention_heads')}\n")

            f.write("\n" + "="*80 + "\n")
            f.write(f"EFFICIENCY STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Pruning Rate: {len(pruned_trials) / self.n_trials * 100:.1f}%\n")
            f.write(f"This saved approximately {len(pruned_trials) * self.max_epochs * 0.5:.0f} epoch-equivalents\n")
            f.write(f"compared to running all trials to completion.\n")

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
        description="Hyperband Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Hyperband optimization for GAT
  python hyperband_optimization.py --model GAT --n_trials 100

  # Run with custom max epochs and reduction factor
  python hyperband_optimization.py --model GCN --n_trials 150 --max_epochs 81 --reduction_factor 3

  # Evaluate on test set after optimization
  python hyperband_optimization.py --model GINE --n_trials 100 --evaluate_test
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of optimization trials (default: 100)')
    parser.add_argument('--max_epochs', type=int, default=81,
                        help='Maximum epochs for training (default: 81)')
    parser.add_argument('--reduction_factor', type=int, default=3,
                        help='Hyperband reduction factor (default: 3)')
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
    optimizer = HyperbandOptimizer(
        model_name=args.model,
        n_trials=args.n_trials,
        max_epochs=args.max_epochs,
        reduction_factor=args.reduction_factor,
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
    print("HYPERBAND OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()