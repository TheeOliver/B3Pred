"""
Grid Search Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Systematically explores a predefined grid of hyperparameter combinations.
Uses a reduced grid to keep computation tractable.
"""

import json
import argparse
import sys
import itertools
from pathlib import Path
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch_geometric.loader import DataLoader

from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
from configs.predictor_config import GraphConfig


# Reduced grid — full Cartesian product would be too large
GRID = {
    'graph_layers':           [2, 3, 5],
    'graph_hidden_channels':  [64, 128, 256],
    'graph_dropouts':         [0.1, 0.3, 0.5],
    'graph_norm':             [True, False],
    'pred_layers':            [2, 3],
    'pred_hidden_channels':   [64, 128],
    'pred_dropouts':          [0.2, 0.4],
    'batch_size':             [32, 64, 128],
    'lr':                     [1e-4, 1e-3, 5e-3],
}

GAT_GRID = {
    'attention_heads':    [2, 4, 8],
    'attention_dropouts': [0.1, 0.3],
}


class GridSearchOptimizer:
    """Grid Search over a predefined hyperparameter grid"""

    def __init__(self, model_name: str, study_name: str = None,
                 results_dir: Path = None, seed: int = 42,
                 opt_subset_size: float = 0.1, opt_epochs: int = 10,
                 top_k: int = 10, full_epochs: int = 100,
                 max_configs: int = None):
        self.model_name = model_name
        self.study_name = study_name or f"{model_name}_gridsearch_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "gridsearch_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.opt_subset_size = opt_subset_size
        self.opt_epochs = opt_epochs
        self.top_k = top_k
        self.full_epochs = full_epochs
        self.max_configs = max_configs

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print("Loading datasets...")
        self.train_df = pd.read_csv(settings.TRAIN_DATA)
        self.val_df   = pd.read_csv(settings.VAL_DATA)
        self.test_df  = pd.read_csv(settings.TEST_DATA)

        self.train_dataset = MoleculeDataset(data=self.train_df)
        self.stats = compute_feature_stats(self.train_dataset)
        normalize_dataset(self.train_dataset, self.stats)

        self.val_dataset = MoleculeDataset(data=self.val_df)
        normalize_dataset(self.val_dataset, self.stats)

        self.test_dataset = MoleculeDataset(data=self.test_df)
        normalize_dataset(self.test_dataset, self.stats)

        if self.opt_subset_size < 1.0:
            n_subset = int(len(self.train_dataset) * self.opt_subset_size)
            generator = torch.Generator().manual_seed(self.seed)
            subset_indices = torch.randperm(len(self.train_dataset), generator=generator)[:n_subset]
            self.train_subset = torch.utils.data.Subset(self.train_dataset, subset_indices)
            print(f"Using {len(self.train_subset)}/{len(self.train_dataset)} samples for optimization ({self.opt_subset_size*100:.1f}%)")
        else:
            self.train_subset = self.train_dataset

        sample = self.train_dataset[0]
        self.graph_info = {
            'node_dim': sample.x.shape[1],
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 4
        }

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")

        self.all_results = []
        self.best_config = None
        self.best_score = -float('inf')

    def _build_grid(self):
        """Build list of all hyperparameter combinations"""
        grid = dict(GRID)
        if self.model_name == 'GAT':
            grid.update(GAT_GRID)

        keys = list(grid.keys())
        values = list(grid.values())
        configs = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            cfg['model_name'] = self.model_name
            cfg['loss'] = 'crossentropy'
            cfg['epochs'] = self.opt_epochs
            cfg['subset_size'] = self.opt_subset_size
            configs.append(cfg)

        random.shuffle(configs)  # shuffle so max_configs sampling is random
        return configs

    def evaluate_config(self, config: dict, trial_idx: int) -> float:
        try:
            train_loader = DataLoader(
                self.train_subset, batch_size=config['batch_size'],
                shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=config['batch_size'],
                shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()
            )

            model_class = GraphConfig.models[self.model_name]['model']
            model = model_class.from_config(config, self.graph_info)

            res, _ = train_model(
                model, train_loader, val_loader, config['epochs'],
                [settings.TARGET_LABEL], loss_type=config['loss'],
                learning_rate=config['lr'], hetero=False, log=False, save_to=None
            )

            val_f1 = res['macro_f1']
            result_entry = {
                'trial': trial_idx,
                'val_f1': float(val_f1),
                'config': config,
                'all_metrics': res
            }
            self.all_results.append(result_entry)

            if val_f1 > self.best_score:
                self.best_score = val_f1
                self.best_config = config
                print(f"\n{'='*60}")
                print(f"NEW BEST: {val_f1:.4f} (trial {trial_idx})")
                print(f"{'='*60}\n")

            return val_f1

        except Exception as e:
            print(f"Trial {trial_idx} failed: {e}")
            import traceback; traceback.print_exc()
            return -1.0

    def run_optimization(self):
        configs = self._build_grid()
        total = len(configs)

        if self.max_configs and self.max_configs < total:
            configs = configs[:self.max_configs]
            total = len(configs)

        print(f"\n{'='*70}")
        print(f"GRID SEARCH FOR {self.model_name}")
        print(f"Total configurations: {total}")
        print(f"Using {self.opt_subset_size*100:.1f}% data for {self.opt_epochs} epochs each")
        print(f"{'='*70}\n")

        for idx, config in enumerate(configs):
            print(f"[{idx+1}/{total}] Evaluating config...")
            score = self.evaluate_config(config, idx)
            print(f"  Val F1: {score:.4f} | Best so far: {self.best_score:.4f}")

        print(f"\n{'='*70}")
        print(f"GRID SEARCH COMPLETE")
        print(f"Best Val F1: {self.best_score:.4f}")
        print(f"{'='*70}\n")

    def retrain_top_k(self, k=None, full_epochs=None):
        if k is None: k = self.top_k
        if full_epochs is None: full_epochs = self.full_epochs

        if not self.all_results:
            print("No results to retrain.")
            return None

        sorted_results = sorted(self.all_results, key=lambda x: x['val_f1'], reverse=True)
        top_configs = [r['config'] for r in sorted_results[:k]]

        print(f"\n{'='*70}")
        print(f"RETRAINING TOP {k} CONFIGURATIONS ON FULL DATASET ({full_epochs} epochs)")
        print(f"{'='*70}\n")

        best_val_f1 = -float('inf')
        best_config = None
        best_model = None
        retrain_results = []

        for i, config in enumerate(top_configs):
            print(f"\n[{i+1}/{k}] Retraining...")
            config = dict(config)
            config['epochs'] = full_epochs
            config['subset_size'] = 1.0

            train_loader = DataLoader(
                self.train_dataset, batch_size=config['batch_size'],
                shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available()
            )
            val_loader = DataLoader(
                self.val_dataset, batch_size=config['batch_size'],
                shuffle=False, num_workers=4, pin_memory=torch.cuda.is_available()
            )

            model_class = GraphConfig.models[self.model_name]['model']
            model = model_class.from_config(config, self.graph_info)

            res, trained_model = train_model(
                model, train_loader, val_loader, config['epochs'],
                [settings.TARGET_LABEL], loss_type=config['loss'],
                learning_rate=config['lr'], hetero=False, log=False, save_to=None
            )

            val_f1 = res['macro_f1']
            print(f"  Val F1 after full training: {val_f1:.4f}")
            retrain_results.append({'config': config, 'val_f1': float(val_f1), 'all_metrics': res})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = config
                best_model = trained_model
                print(f"  → New best among top-{k}!")

        test_loader = DataLoader(self.test_dataset, batch_size=best_config['batch_size'], shuffle=False)
        test_results = test_model(test_loader, best_model, [settings.TEST_LABEL])

        print(f"\nBest val F1 (full training): {best_val_f1:.4f}")
        print(f"Test results:")
        for key, value in test_results.items():
            if value is not None:
                print(f"  {key}: {value:.4f}")

        self.global_best_config = best_config
        self.global_best_fitness = best_val_f1
        self.retrain_results = retrain_results
        self.final_test_results = test_results
        return test_results

    def save_results(self, test_results: dict = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        data = {
            'model_name': self.model_name,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'total_configs_evaluated': len(self.all_results),
            'best_score': float(self.best_score),
            'best_config': self.best_config,
            'test_results': test_results,
            'all_trials': self.all_results,
        }
        if hasattr(self, 'retrain_results'):
            data['retrain_results'] = self.retrain_results
        if hasattr(self, 'final_test_results'):
            data['final_test_results'] = self.final_test_results

        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Detailed results saved to: {results_file}")

        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"GRID SEARCH RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Total configurations evaluated: {len(self.all_results)}\n")
            f.write(f"Best Validation F1: {self.best_score:.4f}\n\n")
            f.write("Best Configuration:\n")
            for key, value in (self.best_config or {}).items():
                f.write(f"  {key:30s}: {value}\n")
            if test_results:
                f.write("\nTest Results:\n")
                for key, value in test_results.items():
                    if value is not None:
                        f.write(f"  {key:30s}: {value:.4f}\n")
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Grid Search Optimization for BBB Prediction Models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'])
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subset_size', type=float, default=0.1)
    parser.add_argument('--opt_epochs', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--full_epochs', type=int, default=100)
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Max number of grid configs to evaluate (default: all)')
    parser.add_argument('--evaluate_test', action='store_true')
    parser.add_argument('--auto_retrain', action='store_true')

    args = parser.parse_args()

    optimizer = GridSearchOptimizer(
        model_name=args.model,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed,
        opt_subset_size=args.subset_size,
        opt_epochs=args.opt_epochs,
        top_k=args.top_k,
        full_epochs=args.full_epochs,
        max_configs=args.max_configs,
    )

    optimizer.run_optimization()

    test_results = None
    if args.auto_retrain:
        test_results = optimizer.retrain_top_k()
    elif args.evaluate_test:
        # just use best config directly
        pass

    optimizer.save_results(test_results)

    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
