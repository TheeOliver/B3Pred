"""
Simulated Annealing Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Probabilistically accepts worse solutions early on (high temperature) to
escape local optima, then gradually focuses on improvement (cooling).
"""

import json
import argparse
import sys
from pathlib import Path
import random
import copy
import math

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


PARAM_SPACE = {
    'graph_layers':          [2, 3, 4, 5],
    'graph_hidden_channels': [32, 64, 128, 256, 512],
    'graph_dropouts':        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'graph_norm':            [True, False],
    'pred_layers':           [2, 3, 4],
    'pred_hidden_channels':  [32, 64, 128, 256],
    'pred_dropouts':         [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    'batch_size':            [16, 32, 64, 128],
    'lr':                    [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
}

GAT_PARAMS = {
    'attention_heads':    [2, 4, 8],
    'attention_dropouts': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
}


class SimulatedAnnealingOptimizer:
    """Simulated Annealing for GNN hyperparameter tuning"""

    def __init__(self, model_name: str, n_iterations: int = 100,
                 initial_temperature: float = 2.0,
                 cooling_rate: float = 0.95,
                 min_temperature: float = 0.01,
                 n_restarts: int = 3,
                 study_name: str = None, results_dir: Path = None,
                 seed: int = 42, opt_subset_size: float = 0.1,
                 opt_epochs: int = 10, top_k: int = 10, full_epochs: int = 100):
        """
        Args:
            n_iterations:        Iterations per restart
            initial_temperature: Starting temperature (higher = more exploration)
            cooling_rate:        Multiplicative factor applied each iteration (0 < rate < 1)
            min_temperature:     Stop cooling below this threshold
            n_restarts:          Number of independent restarts from random configs
        """
        self.model_name = model_name
        self.n_iterations = n_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.n_restarts = n_restarts
        self.study_name = study_name or f"{model_name}_sa_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "sa_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.opt_subset_size = opt_subset_size
        self.opt_epochs = opt_epochs
        self.top_k = top_k
        self.full_epochs = full_epochs

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
            print(f"Using {len(self.train_subset)}/{len(self.train_dataset)} samples ({self.opt_subset_size*100:.1f}%)")
        else:
            self.train_subset = self.train_dataset

        sample = self.train_dataset[0]
        self.graph_info = {
            'node_dim': sample.x.shape[1],
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 4
        }

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")

        self.param_space = dict(PARAM_SPACE)
        if self.model_name == 'GAT':
            self.param_space.update(GAT_PARAMS)

        self.all_results = []
        self.global_best_config = None
        self.global_best_fitness = -float('inf')
        self.evaluation_count = 0
        self.temperature_history = []

    def _random_config(self) -> dict:
        config = {k: random.choice(v) for k, v in self.param_space.items()}
        config['model_name'] = self.model_name
        config['loss'] = 'crossentropy'
        config['epochs'] = self.opt_epochs
        config['subset_size'] = self.opt_subset_size
        return config

    def _perturb(self, config: dict) -> dict:
        """Generate a neighbor by changing one randomly selected parameter"""
        neighbor = copy.deepcopy(config)
        param = random.choice(list(self.param_space.keys()))
        current = neighbor[param]
        choices = [v for v in self.param_space[param] if v != current]
        if choices:
            # Prefer adjacent steps in the ordered parameter space
            param_vals = self.param_space[param]
            if current in param_vals:
                idx = param_vals.index(current)
                adjacent = []
                if idx > 0:
                    adjacent.append(param_vals[idx - 1])
                if idx < len(param_vals) - 1:
                    adjacent.append(param_vals[idx + 1])
                neighbor[param] = random.choice(adjacent if adjacent else choices)
            else:
                neighbor[param] = random.choice(choices)
        return neighbor

    def _acceptance_probability(self, current_fitness: float, candidate_fitness: float,
                                temperature: float) -> float:
        """Boltzmann acceptance: always accept improvements, sometimes accept worse solutions"""
        delta = candidate_fitness - current_fitness
        if delta > 0:
            return 1.0
        if temperature < 1e-10:
            return 0.0
        return math.exp(delta / temperature)

    def _evaluate(self, config: dict) -> float:
        self.evaluation_count += 1
        config = copy.deepcopy(config)
        config['config_name'] = f"{self.study_name}_eval_{self.evaluation_count}"
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

            val_f1 = float(res['macro_f1'])
            self.all_results.append({
                'trial': self.evaluation_count,
                'val_f1': val_f1,
                'config': config,
                'all_metrics': res
            })

            if val_f1 > self.global_best_fitness:
                self.global_best_fitness = val_f1
                self.global_best_config = copy.deepcopy(config)
                print(f"\n{'='*60}\nNEW GLOBAL BEST: {val_f1:.4f} (eval #{self.evaluation_count})\n{'='*60}\n")

            return val_f1

        except Exception as e:
            print(f"Evaluation {self.evaluation_count} failed: {e}")
            import traceback; traceback.print_exc()
            return -1.0

    def _run_single_anneal(self, restart_idx: int):
        """One SA run from a random starting configuration"""
        current_config = self._random_config()
        current_fitness = self._evaluate(current_config)
        temperature = self.initial_temperature

        best_in_run = current_fitness
        best_config_in_run = copy.deepcopy(current_config)

        accepted = 0
        rejected = 0

        print(f"\n  Restart {restart_idx+1}/{self.n_restarts} — "
              f"Start Val F1: {current_fitness:.4f} | T₀={temperature:.3f}")

        for iteration in range(self.n_iterations):
            if temperature < self.min_temperature:
                print(f"  Temperature below minimum ({self.min_temperature}), stopping early.")
                break

            candidate_config = self._perturb(current_config)
            candidate_fitness = self._evaluate(candidate_config)

            prob = self._acceptance_probability(current_fitness, candidate_fitness, temperature)

            if random.random() < prob:
                current_config = candidate_config
                current_fitness = candidate_fitness
                accepted += 1
                if current_fitness > best_in_run:
                    best_in_run = current_fitness
                    best_config_in_run = copy.deepcopy(current_config)
            else:
                rejected += 1

            self.temperature_history.append(temperature)
            temperature = max(self.min_temperature, temperature * self.cooling_rate)

            if (iteration + 1) % 10 == 0:
                acceptance_rate = accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0
                print(f"  Iter {iteration+1}/{self.n_iterations} | "
                      f"T={temperature:.4f} | "
                      f"Current: {current_fitness:.4f} | "
                      f"Run best: {best_in_run:.4f} | "
                      f"Accept rate: {acceptance_rate:.2f}")

        print(f"  Restart {restart_idx+1} done — Best in run: {best_in_run:.4f}")
        return best_config_in_run, best_in_run

    def run_optimization(self):
        print(f"\n{'='*70}")
        print(f"SIMULATED ANNEALING FOR {self.model_name}")
        print(f"Iterations: {self.n_iterations} | Restarts: {self.n_restarts}")
        print(f"T₀={self.initial_temperature} | Cooling rate={self.cooling_rate} | T_min={self.min_temperature}")
        print(f"Data: {self.opt_subset_size*100:.1f}% | Epochs/eval: {self.opt_epochs}")
        print(f"{'='*70}\n")

        for restart in range(self.n_restarts):
            print(f"\n{'='*50}")
            print(f"SA RESTART {restart+1}/{self.n_restarts}")
            print(f"{'='*50}")
            self._run_single_anneal(restart)

        print(f"\n{'='*70}")
        print(f"SA COMPLETE — Global best Val F1: {self.global_best_fitness:.4f}")
        print(f"Total evaluations: {self.evaluation_count}")
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
        print(f"RETRAINING TOP {k} CONFIGS ON FULL DATASET ({full_epochs} epochs)")
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
            print(f"  Val F1: {val_f1:.4f}")
            retrain_results.append({'config': config, 'val_f1': float(val_f1), 'all_metrics': res})

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_config = config
                best_model = trained_model
                print(f"  → New best among top-{k}!")

        test_loader = DataLoader(self.test_dataset, batch_size=best_config['batch_size'], shuffle=False)
        test_results = test_model(test_loader, best_model, [settings.TEST_LABEL])

        print(f"\nBest val F1 (full training): {best_val_f1:.4f}")
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
            'n_iterations': self.n_iterations,
            'n_restarts': self.n_restarts,
            'initial_temperature': self.initial_temperature,
            'cooling_rate': self.cooling_rate,
            'min_temperature': self.min_temperature,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'total_evaluations': self.evaluation_count,
            'best_score': float(self.global_best_fitness),
            'best_config': self.global_best_config,
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
            f.write(f"SIMULATED ANNEALING RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Iterations per restart: {self.n_iterations}\n")
            f.write(f"Restarts: {self.n_restarts}\n")
            f.write(f"Initial temperature: {self.initial_temperature}\n")
            f.write(f"Cooling rate: {self.cooling_rate}\n")
            f.write(f"Min temperature: {self.min_temperature}\n")
            f.write(f"Total evaluations: {self.evaluation_count}\n")
            f.write(f"Best Validation F1: {self.global_best_fitness:.4f}\n\n")
            f.write("Best Configuration:\n")
            for key, value in (self.global_best_config or {}).items():
                f.write(f"  {key:30s}: {value}\n")
            if test_results:
                f.write("\nTest Results:\n")
                for key, value in test_results.items():
                    if value is not None:
                        f.write(f"  {key:30s}: {value:.4f}\n")
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Simulated Annealing Optimization for BBB Prediction Models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'])
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Iterations per restart (default: 50)')
    parser.add_argument('--n_restarts', type=int, default=3,
                        help='Number of restarts (default: 3)')
    parser.add_argument('--initial_temperature', type=float, default=2.0,
                        help='Starting temperature (default: 2.0)')
    parser.add_argument('--cooling_rate', type=float, default=0.95,
                        help='Multiplicative cooling factor per iteration (default: 0.95)')
    parser.add_argument('--min_temperature', type=float, default=0.01,
                        help='Temperature floor below which annealing stops (default: 0.01)')
    parser.add_argument('--study_name', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--subset_size', type=float, default=0.1)
    parser.add_argument('--opt_epochs', type=int, default=10)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--full_epochs', type=int, default=100)
    parser.add_argument('--evaluate_test', action='store_true')
    parser.add_argument('--auto_retrain', action='store_true')

    args = parser.parse_args()

    optimizer = SimulatedAnnealingOptimizer(
        model_name=args.model,
        n_iterations=args.n_iterations,
        n_restarts=args.n_restarts,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        min_temperature=args.min_temperature,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed,
        opt_subset_size=args.subset_size,
        opt_epochs=args.opt_epochs,
        top_k=args.top_k,
        full_epochs=args.full_epochs,
    )

    optimizer.run_optimization()

    test_results = None
    if args.auto_retrain:
        test_results = optimizer.retrain_top_k()

    optimizer.save_results(test_results)

    print("\n" + "="*80)
    print("SIMULATED ANNEALING COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
