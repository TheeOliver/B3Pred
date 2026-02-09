"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimization
for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE
"""

import json
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import cma
from torch_geometric.loader import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from existing modules
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
from configs.predictor_config import GraphConfig


class CMAESOptimizer:
    """CMA-ES Optimization for GNN hyperparameter tuning"""

    def __init__(self, model_name: str, n_iterations: int = 50, population_size: int = None,
                 study_name: str = None, results_dir: Path = None, seed: int = 42):
        """
        Initialize CMA-ES Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_iterations: Number of CMA-ES iterations (generations)
            population_size: Population size (default: 4 + 3*ln(n_params))
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
        """
        self.model_name = model_name
        self.n_iterations = n_iterations
        self.population_size = population_size
        self.study_name = study_name or f"{model_name}_cmaes_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "cmaes_optimization"
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
        self.iteration_count = 0
        self.evaluation_count = 0

        # Define parameter space
        self.param_space = self._define_parameter_space()
        self.n_params = len(self.param_space)

        # Set default population size if not provided
        if self.population_size is None:
            self.population_size = 4 + int(3 * np.log(self.n_params))

    def _define_parameter_space(self) -> list:
        """
        Define parameter space for optimization
        Each parameter is a dict with: name, type, bounds, and options (for categorical)

        Returns:
            List of parameter definitions
        """
        params = [
            # Graph model parameters
            {'name': 'graph_layers', 'type': 'int', 'bounds': [2, 5]},
            {'name': 'graph_hidden_channels', 'type': 'categorical',
             'options': [32, 64, 128, 256, 512]},
            {'name': 'graph_dropouts', 'type': 'float', 'bounds': [0.0, 0.6]},
            {'name': 'graph_norm', 'type': 'binary'},

            # Predictor parameters
            {'name': 'pred_layers', 'type': 'int', 'bounds': [2, 4]},
            {'name': 'pred_hidden_channels', 'type': 'categorical',
             'options': [32, 64, 128, 256]},
            {'name': 'pred_dropouts', 'type': 'float', 'bounds': [0.0, 0.6]},

            # Training parameters
            {'name': 'batch_size', 'type': 'categorical',
             'options': [16, 32, 64, 128]},
            {'name': 'lr', 'type': 'log_float', 'bounds': [1e-5, 1e-2]},
            {'name': 'epochs', 'type': 'int', 'bounds': [20, 100]},
        ]

        # Add model-specific parameters
        if self.model_name == 'GAT':
            params.extend([
                {'name': 'attention_heads', 'type': 'categorical',
                 'options': [2, 4, 8]},
                {'name': 'attention_dropouts', 'type': 'float', 'bounds': [0.0, 0.6]},
            ])

        return params

    def encode_params(self, config: dict) -> np.ndarray:
        """
        Encode configuration dictionary to continuous vector for CMA-ES

        Args:
            config: Configuration dictionary

        Returns:
            Encoded parameter vector
        """
        encoded = []

        for param in self.param_space:
            name = param['name']
            value = config.get(name)

            if param['type'] == 'int':
                # Normalize to [0, 1]
                normalized = (value - param['bounds'][0]) / (param['bounds'][1] - param['bounds'][0])
                encoded.append(normalized)

            elif param['type'] == 'float':
                # Normalize to [0, 1]
                normalized = (value - param['bounds'][0]) / (param['bounds'][1] - param['bounds'][0])
                encoded.append(normalized)

            elif param['type'] == 'log_float':
                # Log scale normalization
                log_min = np.log(param['bounds'][0])
                log_max = np.log(param['bounds'][1])
                log_value = np.log(value)
                normalized = (log_value - log_min) / (log_max - log_min)
                encoded.append(normalized)

            elif param['type'] == 'categorical':
                # One-hot encoding
                idx = param['options'].index(value)
                normalized = idx / (len(param['options']) - 1)
                encoded.append(normalized)

            elif param['type'] == 'binary':
                encoded.append(1.0 if value else 0.0)

        return np.array(encoded)

    def decode_params(self, vector: np.ndarray) -> dict:
        """
        Decode continuous vector to configuration dictionary

        Args:
            vector: Encoded parameter vector

        Returns:
            Configuration dictionary
        """
        config = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_eval_{self.evaluation_count}",
            'loss': 'crossentropy',
            'subset_size': 1.0,
        }

        # Clip vector to [0, 1]
        vector = np.clip(vector, 0.0, 1.0)

        for i, param in enumerate(self.param_space):
            name = param['name']
            value = vector[i]

            if param['type'] == 'int':
                # Denormalize and round
                denormalized = value * (param['bounds'][1] - param['bounds'][0]) + param['bounds'][0]
                config[name] = int(np.round(denormalized))

            elif param['type'] == 'float':
                # Denormalize
                config[name] = value * (param['bounds'][1] - param['bounds'][0]) + param['bounds'][0]

            elif param['type'] == 'log_float':
                # Log scale denormalization
                log_min = np.log(param['bounds'][0])
                log_max = np.log(param['bounds'][1])
                log_value = value * (log_max - log_min) + log_min
                config[name] = np.exp(log_value)

            elif param['type'] == 'categorical':
                # Select closest option
                idx = int(np.round(value * (len(param['options']) - 1)))
                idx = np.clip(idx, 0, len(param['options']) - 1)
                config[name] = param['options'][idx]

            elif param['type'] == 'binary':
                config[name] = value > 0.5

        return config

    def objective(self, vector: np.ndarray) -> float:
        """
        Objective function for CMA-ES (returns negative F1 for minimization)

        Args:
            vector: Encoded parameter vector

        Returns:
            Negative validation F1 score (CMA-ES minimizes)
        """
        self.evaluation_count += 1

        try:
            # Decode parameters
            config = self.decode_params(vector)

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
                'evaluation': self.evaluation_count,
                'iteration': self.iteration_count,
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
                print(f"NEW BEST SCORE: {val_f1:.4f} (Evaluation {self.evaluation_count})")
                print(f"{'='*70}\n")

            # Return negative for minimization
            return -val_f1

        except Exception as e:
            print(f"Evaluation {self.evaluation_count} failed with error: {e}")
            import traceback
            traceback.print_exc()
            return 1.0  # Return bad fitness

    def run_optimization(self):
        """Run CMA-ES optimization"""
        print(f"\n{'='*70}")
        print(f"STARTING CMA-ES OPTIMIZATION FOR {self.model_name}")
        print(f"Number of iterations: {self.n_iterations}")
        print(f"Population size: {self.population_size}")
        print(f"Number of parameters: {self.n_params}")
        print(f"{'='*70}\n")

        # Initialize with reasonable defaults
        initial_config = self._get_default_config()
        x0 = self.encode_params(initial_config)

        # CMA-ES options
        opts = {
            'seed': self.seed,
            'popsize': self.population_size,
            'maxiter': self.n_iterations,
            'bounds': [0.0, 1.0],
            'tolfun': 1e-6,
            'tolx': 1e-6,
            'verbose': 1,
        }

        # Run CMA-ES
        es = cma.CMAEvolutionStrategy(x0, 0.3, opts)

        print("Starting evolution...")
        while not es.stop():
            self.iteration_count = es.countiter

            # Generate and evaluate solutions
            solutions = es.ask()
            fitness_values = [self.objective(x) for x in solutions]
            es.tell(solutions, fitness_values)

            # Log progress
            es.disp()

            print(f"\nIteration {self.iteration_count}/{self.n_iterations}")
            print(f"Best F1 so far: {self.best_score:.4f}")
            print(f"Total evaluations: {self.evaluation_count}")
            print(f"{'='*70}\n")

        # Get final results
        best_vector = es.result.xbest
        best_fitness = es.result.fbest

        print(f"\n{'='*70}")
        print(f"CMA-ES OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total iterations: {es.countiter}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Best validation F1: {-best_fitness:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.best_config.items():
            if key not in ['config_name', 'loss', 'subset_size']:
                print(f"  {key:30s}: {value}")
        print(f"{'='*70}\n")

        return es

    def _get_default_config(self) -> dict:
        """Get default configuration for initialization"""
        config = {
            'model_name': self.model_name,
            'graph_layers': 3,
            'graph_hidden_channels': 128,
            'graph_dropouts': 0.3,
            'graph_norm': True,
            'pred_layers': 2,
            'pred_hidden_channels': 64,
            'pred_dropouts': 0.3,
            'batch_size': 64,
            'lr': 1e-3,
            'epochs': 50,
        }

        if self.model_name == 'GAT':
            config['attention_heads'] = 4
            config['attention_dropouts'] = 0.3

        return config

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

    def save_results(self, es: cma.CMAEvolutionStrategy, test_results: dict = None):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'n_iterations': self.n_iterations,
                'population_size': self.population_size,
                'n_parameters': self.n_params,
                'total_evaluations': self.evaluation_count,
                'best_score': float(self.best_score),
                'best_config': self.best_config,
                'test_results': test_results,
                'parameter_space': self.param_space,
                'all_evaluations': self.all_results
            }, f, indent=2)

        print(f"Detailed results saved to: {results_file}")

        # Save summary to text file with highlighted best results
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"CMA-ES OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Total Iterations: {self.n_iterations}\n")
            f.write(f"Population Size: {self.population_size}\n")
            f.write(f"Number of Parameters: {self.n_params}\n")
            f.write(f"Total Evaluations: {self.evaluation_count}\n")
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

            # Top 10 evaluations
            sorted_results = sorted(self.all_results, key=lambda x: x['val_f1'], reverse=True)
            f.write("="*80 + "\n")
            f.write("TOP 10 EVALUATIONS\n")
            f.write("="*80 + "\n")
            for i, result in enumerate(sorted_results[:10], 1):
                f.write(f"\nRank {i}: Evaluation {result['evaluation']} (Iteration {result['iteration']})\n")
                f.write(f"  Validation F1: {result['val_f1']:.4f}\n")
                f.write(f"  Key hyperparameters:\n")
                config = result['config']
                f.write(f"    graph_layers: {config.get('graph_layers')}\n")
                f.write(f"    graph_hidden_channels: {config.get('graph_hidden_channels')}\n")
                f.write(f"    learning_rate: {config.get('lr'):.6f}\n")
                f.write(f"    batch_size: {config.get('batch_size')}\n")
                if 'attention_heads' in config:
                    f.write(f"    attention_heads: {config.get('attention_heads')}\n")

            # CMA-ES specific statistics
            f.write("\n" + "="*80 + "\n")
            f.write("CMA-ES CONVERGENCE STATISTICS\n")
            f.write("="*80 + "\n")
            f.write(f"Final Sigma: {es.sigma:.6f}\n")
            f.write(f"Condition Number: {es.condition_number:.2e}\n")
            f.write(f"Evaluations per iteration: {self.evaluation_count / es.countiter:.1f}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("="*80 + "\n")

        print(f"Summary saved to: {summary_file}")

        # Save CMA-ES object
        es_file = self.results_dir / f"{self.study_name}_{timestamp}_cmaes.pkl"
        import pickle
        with open(es_file, 'wb') as f:
            pickle.dump(es, f)

        print(f"CMA-ES object saved to: {es_file}")


def main():
    parser = argparse.ArgumentParser(
        description="CMA-ES Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CMA-ES optimization for GAT
  python cmaes_optimization.py --model GAT --n_iterations 50

  # Run with custom population size
  python cmaes_optimization.py --model GCN --n_iterations 100 --population_size 20

  # Evaluate on test set after optimization
  python cmaes_optimization.py --model GINE --n_iterations 50 --evaluate_test
        """
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of CMA-ES iterations (default: 50)')
    parser.add_argument('--population_size', type=int, default=None,
                        help='Population size (default: auto = 4 + 3*ln(n_params))')
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
    optimizer = CMAESOptimizer(
        model_name=args.model,
        n_iterations=args.n_iterations,
        population_size=args.population_size,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed
    )

    # Run optimization
    es = optimizer.run_optimization()

    # Evaluate on test set if requested
    test_results = None
    if args.evaluate_test:
        test_results = optimizer.evaluate_best_model()

    # Save results
    optimizer.save_results(es, test_results)

    print("\n" + "="*80)
    print("CMA-ES OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()