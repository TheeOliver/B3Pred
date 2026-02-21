"""
Particle Swarm Optimization (PSO) for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Uses swarm intelligence with:
- Velocity-based movement in hyperparameter space
- Personal and global best tracking
- Inertia weight decay
- Velocity clamping
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


class Particle:
    """Represents a single particle in the swarm"""
    
    def __init__(self, position: dict, velocity: dict = None):
        """
        Initialize a particle
        
        Args:
            position: Dictionary of hyperparameters (current position)
            velocity: Dictionary of velocities for each hyperparameter
        """
        self.position = position
        self.velocity = velocity if velocity is not None else {}
        self.fitness = None
        self.best_position = copy.deepcopy(position)
        self.best_fitness = -float('inf')
    
    def __repr__(self):
        return f"Particle(fitness={self.fitness:.4f if self.fitness else 'None'}, best={self.best_fitness:.4f})"


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for GNN hyperparameter tuning"""
    
    def __init__(
        self,
        model_name: str,
        n_particles: int = 20,
        n_iterations: int = 50,
        w_start: float = 0.9,
        w_end: float = 0.4,
        c1: float = 2.0,
        c2: float = 2.0,
        v_max_fraction: float = 0.2,
        study_name: str = None,
        results_dir: Path = None,
        seed: int = 42,
        opt_subset_size: float = 0.1,
        opt_epochs: int = 10,
        top_k: int = 10,
        full_epochs: int = 100
    ):
        """
        Initialize Particle Swarm Optimizer
        
        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            n_particles: Number of particles in swarm
            n_iterations: Number of iterations
            w_start: Initial inertia weight
            w_end: Final inertia weight
            c1: Cognitive parameter (personal best attraction)
            c2: Social parameter (global best attraction)
            v_max_fraction: Maximum velocity as fraction of parameter range
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
            opt_subset_size: Fraction of training data to use during optimization
            opt_epochs: Number of epochs to use during optimization
            top_k: Number of top configurations to retrain
            full_epochs: Number of epochs for full training
        """
        self.model_name = model_name
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max_fraction = v_max_fraction
        self.study_name = study_name or f"{model_name}_pso_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "pso_optimization"
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
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 4
        }
        
        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")
        
        # Results tracking
        self.swarm = []
        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        self.history = []
        self.evaluation_count = 0
        
        # Define search space
        self.search_space = self._define_search_space()
        self.param_ranges = self._compute_param_ranges()
    
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
    
    def _compute_param_ranges(self) -> dict:
        """Compute parameter ranges for velocity calculations"""
        ranges = {}
        for param_name, spec in self.search_space.items():
            if spec['type'] in ['int', 'float']:
                ranges[param_name] = spec['max'] - spec['min']
            elif spec['type'] == 'log_float':
                ranges[param_name] = np.log10(spec['max']) - np.log10(spec['min'])
            elif spec['type'] == 'categorical':
                ranges[param_name] = len(spec['values']) - 1
        return ranges
    
    def _random_position(self) -> dict:
        """Generate a random position in search space"""
        position = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_particle_{self.evaluation_count}",
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
    
    def _initialize_velocity(self) -> dict:
        """Initialize random velocity for a particle"""
        velocity = {}
        
        for param_name, param_range in self.param_ranges.items():
            v_max = param_range * self.v_max_fraction
            velocity[param_name] = random.uniform(-v_max, v_max)
        
        return velocity
    
    def _initialize_swarm(self):
        """Initialize the swarm with random particles"""
        print(f"\nInitializing swarm of {self.n_particles} particles...")
        self.swarm = []
        
        for _ in range(self.n_particles):
            position = self._random_position()
            velocity = self._initialize_velocity()
            particle = Particle(position, velocity)
            self.swarm.append(particle)
    
    def _evaluate_particle(self, particle: Particle) -> float:
        """
        Evaluate a particle's fitness
        
        Args:
            particle: Particle to evaluate
            
        Returns:
            Fitness score (validation F1)
        """
        try:
            config = particle.position
            
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
            particle.fitness = fitness
            self.evaluation_count += 1
            
            # Update particle's personal best
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = copy.deepcopy(particle.position)
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = copy.deepcopy(particle.position)
                print(f"\n{'='*70}")
                print(f"NEW GLOBAL BEST: {fitness:.4f}")
                print(f"{'='*70}\n")
            
            return fitness
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            particle.fitness = -1.0
            return -1.0
    
    def _evaluate_swarm(self):
        """Evaluate fitness for all particles"""
        for i, particle in enumerate(self.swarm):
            print(f"Evaluating particle {i+1}/{len(self.swarm)}...", end=" ")
            fitness = self._evaluate_particle(particle)
            print(f"Fitness: {fitness:.4f}")
    
    def _update_velocity(self, particle: Particle, w: float):
        """
        Update particle velocity using PSO equation
        
        v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        
        Args:
            particle: Particle to update
            w: Current inertia weight
        """
        new_velocity = {}
        
        for param_name in self.param_ranges.keys():
            spec = self.search_space[param_name]
            current_vel = particle.velocity[param_name]
            current_pos = particle.position[param_name]
            personal_best = particle.best_position[param_name]
            global_best = self.global_best_position[param_name]
            
            # Handle categorical parameters differently
            if spec['type'] == 'categorical':
                # For categorical, use probability-based movement
                r1, r2 = random.random(), random.random()
                
                # Discrete velocity update
                if personal_best != current_pos:
                    personal_pull = self.c1 * r1
                else:
                    personal_pull = 0.0
                
                if global_best != current_pos:
                    global_pull = self.c2 * r2
                else:
                    global_pull = 0.0
                
                # Combine influences
                total_pull = personal_pull + global_pull
                
                # Decide which value to move toward
                if total_pull > random.random():
                    if personal_pull > global_pull:
                        new_velocity[param_name] = 0  # Will move to personal best
                    else:
                        new_velocity[param_name] = 1  # Will move to global best
                else:
                    new_velocity[param_name] = current_vel
            
            else:
                # Continuous parameters
                r1, r2 = random.random(), random.random()
                
                # Convert to appropriate space
                if spec['type'] == 'log_float':
                    current_pos = np.log10(current_pos)
                    personal_best = np.log10(personal_best)
                    global_best = np.log10(global_best)
                
                # PSO velocity update
                inertia = w * current_vel
                cognitive = self.c1 * r1 * (personal_best - current_pos)
                social = self.c2 * r2 * (global_best - current_pos)
                
                new_vel = inertia + cognitive + social
                
                # Clamp velocity
                v_max = self.param_ranges[param_name] * self.v_max_fraction
                new_vel = np.clip(new_vel, -v_max, v_max)
                
                new_velocity[param_name] = new_vel
        
        particle.velocity = new_velocity
    
    def _update_position(self, particle: Particle):
        """
        Update particle position based on velocity
        
        Args:
            particle: Particle to update
        """
        new_position = copy.deepcopy(particle.position)
        
        for param_name in self.param_ranges.keys():
            spec = self.search_space[param_name]
            velocity = particle.velocity[param_name]
            
            if spec['type'] == 'categorical':
                # Categorical movement
                if velocity == 0:
                    new_position[param_name] = particle.best_position[param_name]
                elif velocity == 1:
                    new_position[param_name] = self.global_best_position[param_name]
                # else keep current value
            
            elif spec['type'] == 'int':
                # Integer parameters
                new_val = particle.position[param_name] + velocity
                new_val = int(np.clip(np.round(new_val), spec['min'], spec['max']))
                new_position[param_name] = new_val
            
            elif spec['type'] == 'float':
                # Float parameters
                new_val = particle.position[param_name] + velocity
                new_val = np.clip(new_val, spec['min'], spec['max'])
                new_position[param_name] = float(new_val)
            
            elif spec['type'] == 'log_float':
                # Log-scale parameters
                current_log = np.log10(particle.position[param_name])
                new_log = current_log + velocity
                log_min = np.log10(spec['min'])
                log_max = np.log10(spec['max'])
                new_log = np.clip(new_log, log_min, log_max)
                new_position[param_name] = float(10 ** new_log)
        
        particle.position = new_position
    
    def _update_iteration(self, iteration: int):
        """
        Update all particles for one iteration
        
        Args:
            iteration: Current iteration number
        """
        # Calculate inertia weight (linear decay)
        w = self.w_start - (self.w_start - self.w_end) * (iteration / self.n_iterations)
        
        # Update velocities and positions
        for particle in self.swarm:
            self._update_velocity(particle, w)
            self._update_position(particle)
        
        # Calculate statistics
        fitnesses = [p.fitness for p in self.swarm if p.fitness is not None]
        avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
        max_fitness = max(fitnesses) if fitnesses else 0.0
        min_fitness = min(fitnesses) if fitnesses else 0.0
        
        print(f"\n{'='*70}")
        print(f"Iteration {iteration}")
        print(f"{'='*70}")
        print(f"Inertia weight: {w:.3f}")
        print(f"Global best:    {self.global_best_fitness:.4f}")
        print(f"Swarm avg:      {avg_fitness:.4f}")
        print(f"Swarm max:      {max_fitness:.4f}")
        print(f"Swarm min:      {min_fitness:.4f}")
        print(f"{'='*70}\n")
        
        # Record history
        self.history.append({
            'iteration': iteration,
            'inertia_weight': float(w),
            'global_best': float(self.global_best_fitness),
            'avg_fitness': float(avg_fitness),
            'max_fitness': float(max_fitness),
            'min_fitness': float(min_fitness),
            'best_config': copy.deepcopy(self.global_best_position)
        })
    
    def run_optimization(self):
        """Run particle swarm optimization"""
        print(f"\n{'='*70}")
        print(f"STARTING PARTICLE SWARM OPTIMIZATION FOR {self.model_name}")
        print(f"Swarm size: {self.n_particles}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Using {self.opt_subset_size*100:.1f}% of data for {self.opt_epochs} epochs per evaluation")
        print(f"Inertia: {self.w_start} -> {self.w_end}")
        print(f"c1 (cognitive): {self.c1}")
        print(f"c2 (social): {self.c2}")
        print(f"{'='*70}\n")
        
        # Initialize swarm
        self._initialize_swarm()
        
        # Evaluate initial swarm
        print("Evaluating initial swarm...")
        self._evaluate_swarm()
        
        # Optimization loop
        for iteration in range(1, self.n_iterations + 1):
            self._update_iteration(iteration)
            self._evaluate_swarm()
        
        # Final statistics
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Global best fitness: {self.global_best_fitness:.4f}")
        print(f"\nBest hyperparameters:")
        for key, value in self.global_best_position.items():
            if key not in ['model_name', 'config_name', 'loss', 'subset_size']:
                print(f"  {key:30s}: {value}")
        print(f"{'='*70}\n")
    
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
            
        # Collect all evaluated configurations from history and current swarm
        all_configs = []
        
        # Add history best configs
        for entry in self.history:
            if entry.get('best_config'):
                all_configs.append({
                    'config': entry['best_config'],
                    'val_f1': entry['global_best']
                })
        
        # Add current swarm particles
        for particle in self.swarm:
            if particle.fitness is not None:
                all_configs.append({
                    'config': particle.position,
                    'val_f1': particle.fitness
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
        self.global_best_position = best_config
        self.global_best_fitness = best_val_f1
        
        # Store retraining results
        self.retrain_results = retrain_results
        self.final_test_results = test_results

        return test_results
    
    def evaluate_best_model(self):
        """Evaluate best model on test set"""
        if self.global_best_position is None:
            print("No best position found. Run optimization first.")
            return None
        
        print(f"\n{'='*70}")
        print(f"EVALUATING BEST MODEL ON TEST SET")
        print(f"{'='*70}\n")
        
        config = self.global_best_position
        
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
        
        print(f"\n{'='*70}")
        print(f"TEST SET RESULTS")
        print(f"{'='*70}")
        for key, value in test_results.items():
            if value is not None:
                print(f"{key:30s}: {value:.4f}")
        print(f"{'='*70}\n")
        
        return test_results
    
    def save_results(self, test_results: dict = None):
        """Save optimization results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data dictionary
        data = {
            'model_name': self.model_name,
            'n_particles': self.n_particles,
            'n_iterations': self.n_iterations,
            'w_start': self.w_start,
            'w_end': self.w_end,
            'c1': self.c1,
            'c2': self.c2,
            'opt_subset_size': self.opt_subset_size,
            'opt_epochs': self.opt_epochs,
            'global_best_fitness': float(self.global_best_fitness),
            'best_config': self.global_best_position,
            'test_results': test_results,
            'history': self.history,
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
            f.write("="*80 + "\n")
            f.write(f"PARTICLE SWARM OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Swarm Size: {self.n_particles}\n")
            f.write(f"Iterations: {self.n_iterations}\n")
            f.write(f"Total Evaluations: {self.evaluation_count}\n")
            f.write(f"Optimization Data Subset: {self.opt_subset_size*100:.1f}%\n")
            f.write(f"Optimization Epochs: {self.opt_epochs}\n")
            f.write(f"Inertia Weight: {self.w_start} -> {self.w_end}\n")
            f.write(f"c1 (cognitive): {self.c1}\n")
            f.write(f"c2 (social): {self.c2}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("="*80 + "\n")
            f.write("*** BEST CONFIGURATION ***\n")
            f.write("="*80 + "\n")
            f.write(f"Global Best Fitness: {self.global_best_fitness:.4f}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in self.global_best_position.items():
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
            
            # Convergence progress
            f.write("="*80 + "\n")
            f.write("CONVERGENCE PROGRESS (Last 10 iterations)\n")
            f.write("="*80 + "\n")
            for entry in self.history[-10:]:
                f.write(f"\nIteration {entry['iteration']}:\n")
                f.write(f"  Global best: {entry['global_best']:.4f}\n")
                f.write(f"  Swarm avg:   {entry['avg_fitness']:.4f}\n")
                f.write(f"  Inertia:     {entry['inertia_weight']:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("="*80 + "\n")
        
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Particle Swarm Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--n_particles', type=int, default=20,
                        help='Number of particles in swarm (default: 20)')
    parser.add_argument('--n_iterations', type=int, default=50,
                        help='Number of iterations (default: 50)')
    parser.add_argument('--w_start', type=float, default=0.9,
                        help='Initial inertia weight (default: 0.9)')
    parser.add_argument('--w_end', type=float, default=0.4,
                        help='Final inertia weight (default: 0.4)')
    parser.add_argument('--c1', type=float, default=2.0,
                        help='Cognitive parameter (default: 2.0)')
    parser.add_argument('--c2', type=float, default=2.0,
                        help='Social parameter (default: 2.0)')
    parser.add_argument('--v_max_fraction', type=float, default=0.2,
                        help='Max velocity as fraction of range (default: 0.2)')
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
    optimizer = ParticleSwarmOptimizer(
        model_name=args.model,
        n_particles=args.n_particles,
        n_iterations=args.n_iterations,
        w_start=args.w_start,
        w_end=args.w_end,
        c1=args.c1,
        c2=args.c2,
        v_max_fraction=args.v_max_fraction,
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
    
    print("\n" + "="*80)
    print("PARTICLE SWARM OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()