"""
Hill Climbing Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Uses local search with:
- Random restart capability
- Multiple neighbor generation strategies
- Simulated annealing option
- Plateau detection and escape
"""

import json
import argparse
import sys
from pathlib import Path
import random
import copy
import math

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch_geometric.loader import DataLoader

# Import from existing modules
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
from configs.predictor_config import GraphConfig


class HillClimbingOptimizer:
    """Hill Climbing with random restarts for GNN hyperparameter tuning"""

    def __init__(
            self,
            model_name: str,
            n_iterations: int = 100,
            n_restarts: int = 5,
            n_neighbors: int = 8,
            neighbor_strategy: str = 'mixed',
            use_simulated_annealing: bool = False,
            initial_temperature: float = 1.0,
            cooling_rate: float = 0.95,
            plateau_patience: int = 5,
            study_name: str = None,
            results_dir: Path = None,
            seed: int = 42,
            opt_subset_size: float = 0.1,
            opt_epochs: int = 10,
            top_k: int = 10,
            full_epochs: int = 100
    ):
        """
        Initialize Hill Climbing Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_iterations: Number of iterations per restart
            n_restarts: Number of random restarts
            n_neighbors: Number of neighbors to generate per iteration
            neighbor_strategy: 'single' (change one param), 'multiple' (change several), or 'mixed'
            use_simulated_annealing: Whether to use simulated annealing acceptance
            initial_temperature: Initial temperature for simulated annealing
            cooling_rate: Cooling rate for simulated annealing
            plateau_patience: Number of iterations without improvement before restart
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
            opt_subset_size: Fraction of training data to use during optimization
            opt_epochs: Number of epochs to use during optimization
            top_k: Number of top configurations to retrain
            full_epochs: Number of epochs for full training
        """
        self.model_name = model_name
        self.n_iterations = n_iterations
        self.n_restarts = n_restarts
        self.n_neighbors = n_neighbors
        self.neighbor_strategy = neighbor_strategy
        self.use_simulated_annealing = use_simulated_annealing
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.plateau_patience = plateau_patience
        self.study_name = study_name or f"{model_name}_hillclimb_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "hillclimbing_optimization"
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
        random.seed(seed)

        # Load datasets
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
            generator = torch.Generator().manual_seed(self.seed)
            subset_indices = torch.randperm(len(self.train_dataset), generator=generator)[:n_subset]
            self.train_subset = torch.utils.data.Subset(self.train_dataset, subset_indices)
            print(f"Using {len(self.train_subset)}/{len(self.train_dataset)} samples for optimization ({self.opt_subset_size*100:.1f}%)")
        else:
            self.train_subset = self.train_dataset
            print("Using full training dataset for optimization")

        # Get graph info
        sample = self.train_dataset[0]
        self.graph_info = {
            'node_dim': sample.x.shape[1],
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample,
                                                             'edge_attr') and sample.edge_attr is not None else 4
        }

        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")

        # Results tracking
        self.global_best_config = None
        self.global_best_fitness = -float('inf')
        self.history = []
        self.evaluation_count = 0
        self.restart_history = []

        # Define search space
        self.search_space = self._define_search_space()

    def _define_search_space(self) -> dict:
        """Define the hyperparameter search space"""
        space = {
            'graph_layers': {'type': 'int', 'min': 2, 'max': 5, 'step': 1},
            'graph_hidden_channels': {'type': 'categorical', 'values': [32, 64, 128, 256, 512]},
            'graph_dropouts': {'type': 'float', 'min': 0.0, 'max': 0.6, 'step': 0.05},
            'graph_norm': {'type': 'categorical', 'values': [True, False]},
            'pred_layers': {'type': 'int', 'min': 2, 'max': 4, 'step': 1},
            'pred_hidden_channels': {'type': 'categorical', 'values': [32, 64, 128, 256]},
            'pred_dropouts': {'type': 'float', 'min': 0.0, 'max': 0.6, 'step': 0.05},
            'batch_size': {'type': 'categorical', 'values': [16, 32, 64, 128]},
            'lr': {'type': 'log_float', 'min': 1e-5, 'max': 1e-2, 'factor': 2.0},
            # Removed: 'epochs': {'type': 'int', 'min': 20, 'max': 100, 'step': 10},
        }

        # Model-specific hyperparameters
        if self.model_name == 'GAT':
            space['attention_heads'] = {'type': 'categorical', 'values': [2, 4, 8]}
            space['attention_dropouts'] = {'type': 'float', 'min': 0.0, 'max': 0.6, 'step': 0.05}

        return space

    def _random_config(self) -> dict:
        """Generate a random configuration"""
        config = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_eval_{self.evaluation_count}",
            'loss': 'crossentropy',
            'subset_size': self.opt_subset_size,
            'epochs': self.opt_epochs,
        }

        for param_name, spec in self.search_space.items():
            if spec['type'] == 'int':
                config[param_name] = random.randint(spec['min'], spec['max'])
            elif spec['type'] == 'float':
                config[param_name] = random.uniform(spec['min'], spec['max'])
            elif spec['type'] == 'log_float':
                log_min = np.log10(spec['min'])
                log_max = np.log10(spec['max'])
                config[param_name] = 10 ** random.uniform(log_min, log_max)
            elif spec['type'] == 'categorical':
                config[param_name] = random.choice(spec['values'])

        return config

    def _generate_neighbor(self, config: dict, n_changes: int = 1) -> dict:
        """
        Generate a neighbor by modifying parameters

        Args:
            config: Current configuration
            n_changes: Number of parameters to change

        Returns:
            Neighboring configuration
        """
        neighbor = copy.deepcopy(config)

        # Select parameters to modify
        modifiable_params = list(self.search_space.keys())
        params_to_change = random.sample(modifiable_params, min(n_changes, len(modifiable_params)))

        for param_name in params_to_change:
            spec = self.search_space[param_name]
            current_val = neighbor[param_name]

            if spec['type'] == 'int':
                # Step up or down
                step = spec.get('step', 1)
                direction = random.choice([-1, 1])
                new_val = current_val + direction * step
                new_val = int(np.clip(new_val, spec['min'], spec['max']))
                neighbor[param_name] = new_val

            elif spec['type'] == 'float':
                # Add Gaussian noise
                step = spec.get('step', (spec['max'] - spec['min']) * 0.1)
                direction = random.choice([-1, 1])
                new_val = current_val + direction * step
                new_val = np.clip(new_val, spec['min'], spec['max'])
                neighbor[param_name] = float(new_val)

            elif spec['type'] == 'log_float':
                # Multiply or divide by factor
                factor = spec.get('factor', 2.0)
                direction = random.choice([-1, 1])
                if direction > 0:
                    new_val = current_val * factor
                else:
                    new_val = current_val / factor
                new_val = np.clip(new_val, spec['min'], spec['max'])
                neighbor[param_name] = float(new_val)

            elif spec['type'] == 'categorical':
                # Choose a different value
                values = [v for v in spec['values'] if v != current_val]
                if values:
                    neighbor[param_name] = random.choice(values)

        return neighbor

    def _generate_neighbors(self, config: dict) -> list:
        """
        Generate multiple neighbors

        Args:
            config: Current configuration

        Returns:
            List of neighboring configurations
        """
        neighbors = []

        for _ in range(self.n_neighbors):
            if self.neighbor_strategy == 'single':
                n_changes = 1
            elif self.neighbor_strategy == 'multiple':
                n_changes = random.randint(2, min(4, len(self.search_space)))
            else:  # mixed
                n_changes = random.choice([1, 1, 2, 3])  # Bias toward single changes

            neighbor = self._generate_neighbor(config, n_changes)
            neighbors.append(neighbor)

        return neighbors

    def _evaluate_config(self, config: dict) -> float:
        """
        Evaluate a configuration

        Args:
            config: Configuration to evaluate

        Returns:
            Fitness score (validation F1)
        """
        try:
            # Create data loaders
            train_loader = DataLoader(
                self.train_subset,
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

            fitness = res['macro_f1']
            self.evaluation_count += 1

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_config = copy.deepcopy(config)
                print(f"\n{'=' * 70}")
                print(f"NEW GLOBAL BEST: {fitness:.4f}")
                print(f"{'=' * 70}\n")

            return fitness

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return -1.0

    def _accept_worse_solution(self, current_fitness: float, neighbor_fitness: float, temperature: float) -> bool:
        """
        Decide whether to accept a worse solution (simulated annealing)

        Args:
            current_fitness: Current solution fitness
            neighbor_fitness: Neighbor solution fitness
            temperature: Current temperature

        Returns:
            True if worse solution should be accepted
        """
        if not self.use_simulated_annealing:
            return False

        if temperature <= 0:
            return False

        delta = neighbor_fitness - current_fitness
        probability = math.exp(delta / temperature)
        return random.random() < probability

    def _hill_climb_restart(self, restart_num: int) -> tuple:
        """
        Perform one hill climbing run with random restart

        Args:
            restart_num: Restart number

        Returns:
            Tuple of (best_config, best_fitness)
        """
        print(f"\n{'=' * 70}")
        print(f"RESTART {restart_num}/{self.n_restarts}")
        print(f"{'=' * 70}\n")

        # Random initialization
        current_config = self._random_config()
        current_fitness = self._evaluate_config(current_config)
        print(f"Initial fitness: {current_fitness:.4f}")

        best_config = copy.deepcopy(current_config)
        best_fitness = current_fitness

        temperature = self.initial_temperature
        plateau_counter = 0

        for iteration in range(1, self.n_iterations + 1):
            # Generate neighbors
            neighbors = self._generate_neighbors(current_config)

            # Evaluate neighbors
            neighbor_fitnesses = []
            for i, neighbor in enumerate(neighbors):
                print(f"Restart {restart_num}, Iteration {iteration}, Neighbor {i + 1}/{len(neighbors)}...", end=" ")
                fitness = self._evaluate_config(neighbor)
                neighbor_fitnesses.append((neighbor, fitness))
                print(f"Fitness: {fitness:.4f}")

            # Find best neighbor
            best_neighbor, best_neighbor_fitness = max(neighbor_fitnesses, key=lambda x: x[1])

            # Decide whether to move to neighbor
            if best_neighbor_fitness > current_fitness:
                # Uphill move
                current_config = best_neighbor
                current_fitness = best_neighbor_fitness
                plateau_counter = 0

                # Update local best
                if current_fitness > best_fitness:
                    best_config = copy.deepcopy(current_config)
                    best_fitness = current_fitness
                    print(f"  ↑ Improved! New best: {best_fitness:.4f}")

            elif self._accept_worse_solution(current_fitness, best_neighbor_fitness, temperature):
                # Accept worse solution (simulated annealing)
                current_config = best_neighbor
                current_fitness = best_neighbor_fitness
                plateau_counter = 0
                print(f"  ↓ Accepted worse solution (SA): {current_fitness:.4f}")

            else:
                # Stuck on plateau
                plateau_counter += 1
                print(f"  → Plateau ({plateau_counter}/{self.plateau_patience})")

                if plateau_counter >= self.plateau_patience:
                    print(f"  Plateau detected! Perturbing solution...")
                    # Large perturbation to escape plateau
                    current_config = self._generate_neighbor(current_config, n_changes=3)
                    current_fitness = self._evaluate_config(current_config)
                    plateau_counter = 0

            # Cool down temperature
            if self.use_simulated_annealing:
                temperature *= self.cooling_rate

            # Record history
            self.history.append({
                'restart': restart_num,
                'iteration': iteration,
                'current_fitness': float(current_fitness),
                'best_fitness': float(best_fitness),
                'temperature': float(temperature) if self.use_simulated_annealing else None,
                'plateau_counter': plateau_counter
            })

        print(f"\nRestart {restart_num} complete. Best fitness: {best_fitness:.4f}")

        self.restart_history.append({
            'restart': restart_num,
            'best_fitness': float(best_fitness),
            'best_config': copy.deepcopy(best_config)
        })

        return best_config, best_fitness

    def run_optimization(self):
        """Run hill climbing optimization with restarts"""
        print(f"\n{'=' * 70}")
        print(f"STARTING HILL CLIMBING OPTIMIZATION FOR {self.model_name}")
        print(f"Iterations per restart: {self.n_iterations}")
        print(f"Number of restarts: {self.n_restarts}")
        print(f"Neighbors per iteration: {self.n_neighbors}")
        print(f"Using {self.opt_subset_size*100:.1f}% of data for {self.opt_epochs} epochs per evaluation")
        print(f"Neighbor strategy: {self.neighbor_strategy}")
        print(f"Simulated annealing: {self.use_simulated_annealing}")
        if self.use_simulated_annealing:
            print(f"Temperature: {self.initial_temperature} (cooling: {self.cooling_rate})")
        print(f"{'=' * 70}\n")

        # Perform multiple restarts
        for restart in range(1, self.n_restarts + 1):
            self._hill_climb_restart(restart)

        # Final statistics
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Global best fitness: {self.global_best_fitness:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.global_best_config.items():
            if key not in ['model_name', 'config_name', 'loss', 'subset_size']:
                print(f"  {key:30s}: {value}")
        print(f"{'=' * 70}\n")

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
            
        # Collect all evaluated configurations from restart history and global best
        all_configs = []
        
        # Add restart history best configs
        for entry in self.restart_history:
            all_configs.append({
                'config': entry['best_config'],
                'val_f1': entry['best_fitness']
            })
        
        # Add global best if not already included
        if self.global_best_config and self.global_best_config not in [c['config'] for c in all_configs]:
            all_configs.append({
                'config': self.global_best_config,
                'val_f1': self.global_best_fitness
            })

        if not all_configs:
            print("No results available to retrain. Run optimization first.")
            return None

        # Sort by validation F1 (descending)
        sorted_results = sorted(all_configs, key=lambda x: x['val_f1'], reverse=True)
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
                self.train_dataset,
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
                'original_config': config.get('config_name', f'config_{i}'),
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
            shuffle=False,
            num_workers=0
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
        if self.global_best_config is None:
            print("No best configuration found. Run optimization first.")
            return None

        print(f"\n{'=' * 70}")
        print(f"EVALUATING BEST MODEL ON TEST SET")
        print(f"{'=' * 70}\n")

        config = self.global_best_config

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Build and train model
        model_class = GraphConfig.models[self.model_name]['model']
        model = model_class.from_config(config, self.graph_info)

        _, trained_model = train_model(
            model,
            train_loader,
            test_loader,
            config['epochs'],
            [settings.TARGET_LABEL],
            loss_type=config['loss'],
            learning_rate=config['lr'],
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

        print(f"\n{'=' * 70}")
        print(f"TEST SET RESULTS")
        print(f"{'=' * 70}")
        for key, value in test_results.items():
            if value is not None:
                print(f"{key:30s}: {value:.4f}")
        print(f"{'=' * 70}\n")

        return test_results

    def save_results(self, test_results: dict = None):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data dictionary
        data = {
            'model_name': self.model_name,
            'n_iterations': self.n_iterations,
            'n_restarts': self.n_restarts,
            'n_neighbors': self.n_neighbors,
            'neighbor_strategy': self.neighbor_strategy,
            'use_simulated_annealing': self.use_simulated_annealing,
            'initial_temperature': self.initial_temperature,
            'cooling_rate': self.cooling_rate,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'global_best_fitness': float(self.global_best_fitness),
            'best_config': self.global_best_config,
            'test_results': test_results,
            'history': self.history,
            'restart_history': self.restart_history,
            'total_evaluations': self.evaluation_count
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

        # Save summary
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"HILL CLIMBING OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Iterations per restart: {self.n_iterations}\n")
            f.write(f"Number of restarts: {self.n_restarts}\n")
            f.write(f"Total Evaluations: {self.evaluation_count}\n")
            f.write(f"Optimization Data Subset: {self.opt_subset_size*100:.1f}%\n")
            f.write(f"Optimization Epochs: {self.opt_epochs}\n")
            f.write(f"Neighbors per iteration: {self.n_neighbors}\n")
            f.write(f"Neighbor strategy: {self.neighbor_strategy}\n")
            f.write(f"Simulated annealing: {self.use_simulated_annealing}\n")
            if self.use_simulated_annealing:
                f.write(f"Initial temperature: {self.initial_temperature}\n")
                f.write(f"Cooling rate: {self.cooling_rate}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            f.write("=" * 80 + "\n")
            f.write("*** BEST CONFIGURATION ***\n")
            f.write("=" * 80 + "\n")
            f.write(f"Global Best Fitness: {self.global_best_fitness:.4f}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in self.global_best_config.items():
                f.write(f"  {key:30s}: {value}\n")
            f.write("\n")

            if test_results:
                f.write("=" * 80 + "\n")
                f.write("*** TEST SET PERFORMANCE ***\n")
                f.write("=" * 80 + "\n")
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

            # Restart summary
            f.write("=" * 80 + "\n")
            f.write("RESTART SUMMARY\n")
            f.write("=" * 80 + "\n")
            for entry in self.restart_history:
                f.write(f"\nRestart {entry['restart']}:\n")
                f.write(f"  Best fitness: {entry['best_fitness']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("=" * 80 + "\n")

        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Hill Climbing Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--n_iterations', type=int, default=100,
                        help='Iterations per restart (default: 100)')
    parser.add_argument('--n_restarts', type=int, default=5,
                        help='Number of random restarts (default: 5)')
    parser.add_argument('--n_neighbors', type=int, default=8,
                        help='Neighbors per iteration (default: 8)')
    parser.add_argument('--neighbor_strategy', type=str, default='mixed',
                        choices=['single', 'multiple', 'mixed'],
                        help='Neighbor generation strategy (default: mixed)')
    parser.add_argument('--use_simulated_annealing', action='store_true',
                        help='Use simulated annealing')
    parser.add_argument('--initial_temperature', type=float, default=1.0,
                        help='Initial temperature for SA (default: 1.0)')
    parser.add_argument('--cooling_rate', type=float, default=0.95,
                        help='Cooling rate for SA (default: 0.95)')
    parser.add_argument('--plateau_patience', type=int, default=5,
                        help='Plateau patience before perturbation (default: 5)')
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
    optimizer = HillClimbingOptimizer(
        model_name=args.model,
        n_iterations=args.n_iterations,
        n_restarts=args.n_restarts,
        n_neighbors=args.n_neighbors,
        neighbor_strategy=args.neighbor_strategy,
        use_simulated_annealing=args.use_simulated_annealing,
        initial_temperature=args.initial_temperature,
        cooling_rate=args.cooling_rate,
        plateau_patience=args.plateau_patience,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed,
        opt_subset_size=args.subset_size,
        opt_epochs=args.opt_epochs,
        top_k=args.top_k,
        full_epochs=args.full_epochs
    )

    # Run optimization
    optimizer.run_optimization()

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
    optimizer.save_results(test_results)

    print("\n" + "=" * 80)
    print("HILL CLIMBING OPTIMIZATION COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()