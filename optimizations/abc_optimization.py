"""
Artificial Bee Colony (ABC) Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Uses bee colony foraging behavior with:
- Employed bees (exploit current food sources)
- Onlooker bees (probabilistic selection based on fitness)
- Scout bees (explore new food sources when exhausted)
- Abandonment mechanism for poor solutions
"""

import json
import argparse
import sys
from pathlib import Path
import random
import copy

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


class FoodSource:
    """Represents a food source (hyperparameter configuration) in ABC"""

    def __init__(self, position: dict, fitness: float = None, trials: int = 0):
        """
        Initialize a food source

        Args:
            position: Dictionary of hyperparameters
            fitness: Fitness score (validation F1)
            trials: Number of trials without improvement
        """
        self.position = position
        self.fitness = fitness
        self.trials = trials  # Counter for abandonment

    def __repr__(self):
        return f"FoodSource(fitness={self.fitness:.4f if self.fitness else 'None'}, trials={self.trials})"


class ArtificialBeeColonyOptimizer:
    """Artificial Bee Colony Optimization for GNN hyperparameter tuning"""

    def __init__(
            self,
            model_name: str,
            colony_size: int = 20,
            n_iterations: int = 50,
            limit: int = 10,
            study_name: str = None,
            results_dir: Path = None,
            seed: int = 42,
            opt_subset_size: float = 0.1,
            opt_epochs: int = 10,
            top_k: int = 10,
            full_epochs: int = 100
    ):
        """
        Initialize Artificial Bee Colony Optimizer

        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            colony_size: Number of food sources (employed bees)
            n_iterations: Number of iterations
            limit: Abandonment limit (trials before becoming scout)
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
            opt_subset_size: Fraction of training data to use during optimization
            opt_epochs: Number of epochs to use during optimization
            top_k: Number of top configurations to retrain
            full_epochs: Number of epochs for full training
        """
        self.model_name = model_name
        self.colony_size = colony_size
        self.n_iterations = n_iterations
        self.limit = limit
        self.study_name = study_name or f"{model_name}_abc_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "abc_optimization"
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
        self.food_sources = []
        self.global_best_source = None
        self.global_best_fitness = -float('inf')
        self.history = []
        self.evaluation_count = 0

        # Define search space
        self.search_space = self._define_search_space()

    def _define_search_space(self) -> dict:
        """Define the hyperparameter search space"""
        space = {
            'graph_layers': {'type': 'int', 'min': 2, 'max': 5},
            'graph_hidden_channels': {'type': 'categorical', 'values': [32, 64, 128, 256, 512]},
            'graph_dropouts': {'type': 'float', 'min': 0.0, 'max': 0.6},
            'graph_norm': {'type': 'categorical', 'values': [True, False]},
            'pred_layers': {'type': 'int', 'min': 2, 'max': 4},
            'pred_hidden_channels': {'type': 'categorical', 'values': [32, 64, 128, 256]},
            'pred_dropouts': {'type': 'float', 'min': 0.0, 'max': 0.6},
            'batch_size': {'type': 'categorical', 'values': [16, 32, 64, 128]},
            'lr': {'type': 'log_float', 'min': 1e-5, 'max': 1e-2},
            # Removed: 'epochs': {'type': 'int', 'min': 20, 'max': 100},
        }

        # Model-specific hyperparameters
        if self.model_name == 'GAT':
            space['attention_heads'] = {'type': 'categorical', 'values': [2, 4, 8]}
            space['attention_dropouts'] = {'type': 'float', 'min': 0.0, 'max': 0.6}

        return space

    def _random_position(self) -> dict:
        """Generate a random position in search space"""
        position = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_food_{self.evaluation_count}",
            'loss': 'crossentropy',
            'subset_size': self.opt_subset_size,
            'epochs': self.opt_epochs,
        }

        for param_name, spec in self.search_space.items():
            if spec['type'] == 'int':
                position[param_name] = random.randint(spec['min'], spec['max'])
            elif spec['type'] == 'float':
                position[param_name] = random.uniform(spec['min'], spec['max'])
            elif spec['type'] == 'log_float':
                log_min = np.log10(spec['min'])
                log_max = np.log10(spec['max'])
                position[param_name] = 10 ** random.uniform(log_min, log_max)
            elif spec['type'] == 'categorical':
                position[param_name] = random.choice(spec['values'])

        return position

    def _initialize_food_sources(self):
        """Initialize food sources randomly"""
        print(f"\nInitializing {self.colony_size} food sources...")
        self.food_sources = []

        for _ in range(self.colony_size):
            position = self._random_position()
            food_source = FoodSource(position)
            self.food_sources.append(food_source)

    def _evaluate_food_source(self, food_source: FoodSource) -> float:
        """
        Evaluate a food source's fitness

        Args:
            food_source: FoodSource to evaluate

        Returns:
            Fitness score (validation F1)
        """
        try:
            config = food_source.position

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
                self.opt_epochs,
                [settings.TARGET_LABEL],
                loss_type=config['loss'],
                learning_rate=config['lr'],
                hetero=False,
                log=False,
                save_to=None
            )

            fitness = res['macro_f1']
            food_source.fitness = fitness
            self.evaluation_count += 1

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_source = copy.deepcopy(food_source)
                print(f"\n{'=' * 70}")
                print(f"NEW GLOBAL BEST: {fitness:.4f}")
                print(f"{'=' * 70}\n")

            return fitness

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            food_source.fitness = -1.0
            return -1.0

    def _generate_neighbor(self, food_source: FoodSource, partner: FoodSource) -> dict:
        """
        Generate a neighbor position using ABC equation

        v_ij = x_ij + phi_ij * (x_ij - x_kj)
        where k is a randomly chosen food source different from i

        Args:
            food_source: Current food source
            partner: Random partner food source for comparison

        Returns:
            New position (neighbor)
        """
        new_position = copy.deepcopy(food_source.position)

        # Select a random parameter to modify
        modifiable_params = list(self.search_space.keys())
        param_to_modify = random.choice(modifiable_params)

        spec = self.search_space[param_to_modify]
        current_val = food_source.position[param_to_modify]
        partner_val = partner.position[param_to_modify]

        if spec['type'] == 'categorical':
            # For categorical: probabilistically choose between current and partner
            if random.random() < 0.5:
                new_position[param_to_modify] = partner_val
            # else keep current

        elif spec['type'] == 'int':
            # ABC equation for integers
            phi = random.uniform(-1, 1)
            new_val = current_val + phi * (current_val - partner_val)
            new_val = int(np.clip(np.round(new_val), spec['min'], spec['max']))
            new_position[param_to_modify] = new_val

        elif spec['type'] == 'float':
            # ABC equation for floats
            phi = random.uniform(-1, 1)
            new_val = current_val + phi * (current_val - partner_val)
            new_val = np.clip(new_val, spec['min'], spec['max'])
            new_position[param_to_modify] = float(new_val)

        elif spec['type'] == 'log_float':
            # ABC equation in log space
            phi = random.uniform(-1, 1)
            current_log = np.log10(current_val)
            partner_log = np.log10(partner_val)
            new_log = current_log + phi * (current_log - partner_log)
            log_min = np.log10(spec['min'])
            log_max = np.log10(spec['max'])
            new_log = np.clip(new_log, log_min, log_max)
            new_position[param_to_modify] = float(10 ** new_log)

        return new_position

    def _employed_bee_phase(self):
        """Employed bees exploit their food sources"""
        print("\n--- Employed Bee Phase ---")

        for i, food_source in enumerate(self.food_sources):
            print(f"Employed bee {i + 1}/{len(self.food_sources)}...", end=" ")

            # Select random partner
            partner_idx = random.choice([j for j in range(len(self.food_sources)) if j != i])
            partner = self.food_sources[partner_idx]

            # Generate neighbor
            new_position = self._generate_neighbor(food_source, partner)
            candidate = FoodSource(new_position)

            # Evaluate
            new_fitness = self._evaluate_food_source(candidate)
            # Just avoid the ternary in the f-string:
            if food_source.fitness is not None:
                print(f"Employed bee {i+1}/{len(self.food_sources)}... Current: {food_source.fitness:.4f}, Candidate: {new_fitness:.4f}")
            else:
                print(f"Employed bee {i+1}/{len(self.food_sources)}... Current: None, Candidate: {new_fitness:.4f}")

            # Greedy selection
            if food_source.fitness is None or new_fitness > food_source.fitness:
                self.food_sources[i] = candidate
                self.food_sources[i].trials = 0
                print(f"  → Improved!")
            else:
                self.food_sources[i].trials += 1
                print(f"  → Rejected (trials: {self.food_sources[i].trials})")

    def _calculate_selection_probabilities(self):
        """Calculate selection probabilities for onlooker bees"""
        fitnesses = np.array([fs.fitness if fs.fitness is not None else 0.0 for fs in self.food_sources])

        # Shift to positive (add minimum if negative)
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min()

        # Avoid division by zero
        total_fitness = fitnesses.sum()
        if total_fitness == 0:
            probabilities = np.ones(len(self.food_sources)) / len(self.food_sources)
        else:
            probabilities = fitnesses / total_fitness

        return probabilities

    def _onlooker_bee_phase(self):
        """Onlooker bees select food sources probabilistically"""
        print("\n--- Onlooker Bee Phase ---")

        probabilities = self._calculate_selection_probabilities()

        if len(probabilities) != len(self.food_sources):
            print(f"WARNING: probabilities length {len(probabilities)} != food_sources length {len(self.food_sources)}")
            return

        for i in range(self.colony_size):
            print(f"Onlooker bee {i + 1}/{self.colony_size}...", end=" ")

            # Select food source based on probability
            selected_idx = np.random.choice(len(self.food_sources), p=probabilities)
            food_source = self.food_sources[selected_idx]

            # Select random partner
            partner_idx = random.choice([j for j in range(len(self.food_sources)) if j != selected_idx])
            partner = self.food_sources[partner_idx]

            # Generate neighbor
            new_position = self._generate_neighbor(food_source, partner)
            candidate = FoodSource(new_position)

            # Evaluate
            new_fitness = self._evaluate_food_source(candidate)
            print(f"Selected: {selected_idx}, Current: {food_source.fitness:.4f}, Candidate: {new_fitness:.4f}")

            # Greedy selection
            if new_fitness > food_source.fitness:
                self.food_sources[selected_idx] = candidate
                self.food_sources[selected_idx].trials = 0
                print(f"  → Improved!")
            else:
                self.food_sources[selected_idx].trials += 1
                print(f"  → Rejected (trials: {self.food_sources[selected_idx].trials})")

    def _scout_bee_phase(self):
        """Scout bees replace abandoned food sources"""
        print("\n--- Scout Bee Phase ---")

        abandoned_count = 0
        for i, food_source in enumerate(self.food_sources):
            if food_source.trials >= self.limit:
                print(f"Food source {i} abandoned (trials: {food_source.trials})")
                # Replace with new random position
                new_position = self._random_position()
                self.food_sources[i] = FoodSource(new_position)
                self._evaluate_food_source(self.food_sources[i])
                abandoned_count += 1

        if abandoned_count == 0:
            print("No food sources abandoned")
        else:
            print(f"Replaced {abandoned_count} abandoned food source(s)")

    def _iteration_summary(self, iteration: int):
        """Print iteration summary"""
        fitnesses = [fs.fitness for fs in self.food_sources if fs.fitness is not None]
        avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
        max_fitness = max(fitnesses) if fitnesses else 0.0
        min_fitness = min(fitnesses) if fitnesses else 0.0

        print(f"\n{'=' * 70}")
        print(f"Iteration {iteration} Summary")
        print(f"{'=' * 70}")
        print(f"Global Best:  {self.global_best_fitness:.4f}")
        print(f"Colony Avg:   {avg_fitness:.4f}")
        print(f"Colony Max:   {max_fitness:.4f}")
        print(f"Colony Min:   {min_fitness:.4f}")
        print(f"Evaluations:  {self.evaluation_count}")
        print(f"{'=' * 70}\n")

        # Record history
        self.history.append({
            'iteration': iteration,
            'global_best': float(self.global_best_fitness),
            'avg_fitness': float(avg_fitness),
            'max_fitness': float(max_fitness),
            'min_fitness': float(min_fitness),
            'evaluations': self.evaluation_count
        })

    def run_optimization(self):
        """Run Artificial Bee Colony optimization"""
        print(f"\n{'=' * 70}")
        print(f"STARTING ARTIFICIAL BEE COLONY OPTIMIZATION FOR {self.model_name}")
        print(f"Colony size: {self.colony_size}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Using {self.opt_subset_size*100:.1f}% of data for {self.opt_epochs} epochs per evaluation")
        print(f"Abandonment limit: {self.limit}")
        print(f"{'=' * 70}\n")

        # Initialize colony
        self._initialize_food_sources()

        # Evaluate initial food sources
        print("Evaluating initial food sources...")
        for i, food_source in enumerate(self.food_sources):
            print(f"Initial food source {i + 1}/{len(self.food_sources)}...", end=" ")
            fitness = self._evaluate_food_source(food_source)
            print(f"Fitness: {fitness:.4f}")

        # Main ABC loop
        for iteration in range(1, self.n_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}/{self.n_iterations}")
            print(f"{'=' * 70}")

            # Three phases of ABC
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()

            # Summary
            self._iteration_summary(iteration)

        # Final statistics
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Global best fitness: {self.global_best_fitness:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.global_best_source.position.items():
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
            
        # Collect all evaluated configurations from food sources and history
        all_configs = []
        
        # Add current food sources
        for fs in self.food_sources:
            if fs.fitness is not None:
                all_configs.append({
                    'config': fs.position,
                    'val_f1': fs.fitness
                })
        
        # Add global best if not already included
        if self.global_best_source and self.global_best_source.position not in [c['config'] for c in all_configs]:
            all_configs.append({
                'config': self.global_best_source.position,
                'val_f1': self.global_best_source.fitness
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
        self.global_best_source = FoodSource(best_config, best_val_f1)
        self.global_best_fitness = best_val_f1
        
        # Store retraining results
        self.retrain_results = retrain_results
        self.final_test_results = test_results

        return test_results

    def evaluate_best_model(self):
        """Evaluate best model on test set"""
        if self.global_best_source is None:
            print("No best solution found. Run optimization first.")
            return None

        print(f"\n{'=' * 70}")
        print(f"EVALUATING BEST MODEL ON TEST SET")
        print(f"{'=' * 70}\n")

        config = self.global_best_source.position

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
        
        # Helper function to convert numpy types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj

        # Prepare data dictionary with conversion
        data = {
            'model_name': self.model_name,
            'colony_size': self.colony_size,
            'n_iterations': self.n_iterations,
            'limit': self.limit,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'best_score': float(self.global_best_fitness),
            'best_config': convert_numpy(self.global_best_source.position) if self.global_best_source else None,
            'test_results': convert_numpy(test_results),
            'history': convert_numpy(self.history),
            'total_evaluations': self.evaluation_count
        }
        
        # Add retraining results if available
        if hasattr(self, 'retrain_results'):
            data['retrain_results'] = convert_numpy(self.retrain_results)
        if hasattr(self, 'final_test_results'):
            data['final_test_results'] = convert_numpy(self.final_test_results)

        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Detailed results saved to: {results_file}")

        # Save summary
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ARTIFICIAL BEE COLONY OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Colony Size: {self.colony_size}\n")
            f.write(f"Iterations: {self.n_iterations}\n")
            f.write(f"Total Evaluations: {self.evaluation_count}\n")
            f.write(f"Optimization Data Subset: {self.opt_subset_size*100:.1f}%\n")
            f.write(f"Optimization Epochs: {self.opt_epochs}\n")
            f.write(f"Abandonment Limit: {self.limit}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            f.write("=" * 80 + "\n")
            f.write("*** BEST CONFIGURATION ***\n")
            f.write("=" * 80 + "\n")
            f.write(f"Global Best Fitness: {self.global_best_fitness:.4f}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in self.global_best_source.position.items():
                if key not in ['model_name', 'config_name', 'loss', 'subset_size']:
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

            # Convergence progress
            f.write("=" * 80 + "\n")
            f.write("CONVERGENCE PROGRESS (Last 10 iterations)\n")
            f.write("=" * 80 + "\n")
            for entry in self.history[-10:]:
                f.write(f"\nIteration {entry['iteration']}:\n")
                f.write(f"  Global best: {entry['global_best']:.4f}\n")
                f.write(f"  Colony avg:  {entry['avg_fitness']:.4f}\n")
                f.write(f"  Evaluations: {entry['evaluations']}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("=" * 80 + "\n")

        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Artificial Bee Colony Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--colony_size', type=int, default=20,
                        help='Number of food sources (default: 20)')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of iterations (default: 50)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Abandonment limit (default: 10)')
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
    optimizer = ArtificialBeeColonyOptimizer(
        model_name=args.model,
        colony_size=args.colony_size,
        n_iterations=args.n_iterations,
        limit=args.limit,
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
    print("ARTIFICIAL BEE COLONY OPTIMIZATION COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()