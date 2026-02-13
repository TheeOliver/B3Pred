"""
Genetic Algorithm Optimization for BBB Prediction Models
Supports all GNN architectures: GAT, GCN, GraphSAGE, GIN, GINE

Uses evolutionary strategies with:
- Tournament selection
- Uniform and arithmetic crossover
- Gaussian mutation
- Elitism
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


class Individual:
    """Represents a single individual (hyperparameter configuration) in the population"""
    
    def __init__(self, genes: dict, fitness: float = None):
        """
        Initialize an individual
        
        Args:
            genes: Dictionary of hyperparameters
            fitness: Fitness score (validation F1)
        """
        self.genes = genes
        self.fitness = fitness
        self.generation = 0
    
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.4f if self.fitness else 'None'})"


class GeneticOptimizer:
    """Genetic Algorithm for GNN hyperparameter tuning"""
    
    def __init__(
        self,
        model_name: str,
        population_size: int = 20,
        n_generations: int = 50,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.8,
        elitism_rate: float = 0.1,
        tournament_size: int = 3,
        study_name: str = None,
        results_dir: Path = None,
        seed: int = 42
    ):
        """
        Initialize Genetic Algorithm Optimizer
        
        Args:
            model_name: One of ['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE']
            population_size: Number of individuals in population
            n_generations: Number of generations to evolve
            mutation_rate: Probability of mutation per gene
            crossover_rate: Probability of crossover
            elitism_rate: Fraction of top individuals to preserve
            tournament_size: Size of tournament for selection
            study_name: Name for the optimization study
            results_dir: Directory to save results
            seed: Random seed
        """
        self.model_name = model_name
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.tournament_size = tournament_size
        self.study_name = study_name or f"{model_name}_genetic_opt"
        self.results_dir = results_dir or settings.EXPERIMENTS_FOLDER / "genetic_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
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
        
        # Get graph info
        sample = self.train_dataset[0]
        self.graph_info = {
            'node_dim': sample.x.shape[1],
            'edge_dim': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 4
        }
        
        print(f"Train: {len(self.train_dataset)} | Val: {len(self.val_dataset)} | Test: {len(self.test_dataset)}")
        print(f"Node dim: {self.graph_info['node_dim']}, Edge dim: {self.graph_info['edge_dim']}")
        
        # Results tracking
        self.population = []
        self.history = []
        self.best_individual = None
        self.best_score = -float('inf')
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
            'epochs': {'type': 'int', 'min': 20, 'max': 100},
        }
        
        # Model-specific hyperparameters
        if self.model_name == 'GAT':
            space['attention_heads'] = {'type': 'categorical', 'values': [2, 4, 8]}
            space['attention_dropouts'] = {'type': 'float', 'min': 0.0, 'max': 0.6}
        
        return space
    
    def _random_gene_value(self, gene_name: str):
        """Generate a random value for a gene based on its type"""
        spec = self.search_space[gene_name]
        
        if spec['type'] == 'int':
            return random.randint(spec['min'], spec['max'])
        elif spec['type'] == 'float':
            return random.uniform(spec['min'], spec['max'])
        elif spec['type'] == 'log_float':
            log_min = np.log10(spec['min'])
            log_max = np.log10(spec['max'])
            return 10 ** random.uniform(log_min, log_max)
        elif spec['type'] == 'categorical':
            return random.choice(spec['values'])
        else:
            raise ValueError(f"Unknown gene type: {spec['type']}")
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual"""
        genes = {
            'model_name': self.model_name,
            'config_name': f"{self.study_name}_ind_{self.evaluation_count}",
            'loss': 'crossentropy',
            'subset_size': 1.0,
        }
        
        # Generate random values for all genes
        for gene_name in self.search_space.keys():
            genes[gene_name] = self._random_gene_value(gene_name)
        
        return Individual(genes)
    
    def _initialize_population(self):
        """Initialize the population with random individuals"""
        print(f"\nInitializing population of size {self.population_size}...")
        self.population = [self._create_random_individual() for _ in range(self.population_size)]
    
    def _evaluate_individual(self, individual: Individual) -> float:
        """
        Evaluate an individual's fitness
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Fitness score (validation F1)
        """
        try:
            config = individual.genes
            
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
            
            fitness = res['macro_f1']
            individual.fitness = fitness
            self.evaluation_count += 1
            
            # Update best individual
            if fitness > self.best_score:
                self.best_score = fitness
                self.best_individual = copy.deepcopy(individual)
                print(f"\n{'='*70}")
                print(f"NEW BEST SCORE: {fitness:.4f} (Generation {individual.generation})")
                print(f"{'='*70}\n")
            
            return fitness
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            individual.fitness = -1.0
            return -1.0
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals in population"""
        for i, individual in enumerate(self.population):
            if individual.fitness is None:
                print(f"Evaluating individual {i+1}/{len(self.population)}...", end=" ")
                fitness = self._evaluate_individual(individual)
                print(f"Fitness: {fitness:.4f}")
    
    def _tournament_selection(self) -> Individual:
        """Select an individual using tournament selection"""
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda ind: ind.fitness if ind.fitness else -1.0)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> tuple:
        """
        Perform crossover between two parents
        
        Uses uniform crossover for categorical/discrete genes
        and arithmetic crossover for continuous genes
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Tuple of two offspring
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        # Create offspring
        genes1 = copy.deepcopy(parent1.genes)
        genes2 = copy.deepcopy(parent2.genes)
        
        # Perform crossover for each gene
        for gene_name in self.search_space.keys():
            spec = self.search_space[gene_name]
            
            if spec['type'] == 'categorical' or spec['type'] == 'int':
                # Uniform crossover
                if random.random() < 0.5:
                    genes1[gene_name], genes2[gene_name] = genes2[gene_name], genes1[gene_name]
            else:
                # Arithmetic crossover for continuous values
                alpha = random.random()
                val1 = genes1[gene_name]
                val2 = genes2[gene_name]
                genes1[gene_name] = alpha * val1 + (1 - alpha) * val2
                genes2[gene_name] = (1 - alpha) * val1 + alpha * val2
        
        offspring1 = Individual(genes1)
        offspring2 = Individual(genes2)
        
        return offspring1, offspring2
    
    def _mutate(self, individual: Individual):
        """
        Mutate an individual's genes
        
        Args:
            individual: Individual to mutate (modified in-place)
        """
        for gene_name in self.search_space.keys():
            if random.random() < self.mutation_rate:
                spec = self.search_space[gene_name]
                
                if spec['type'] == 'categorical':
                    # Random replacement
                    individual.genes[gene_name] = random.choice(spec['values'])
                
                elif spec['type'] == 'int':
                    # Gaussian mutation with bounds
                    current = individual.genes[gene_name]
                    sigma = (spec['max'] - spec['min']) * 0.2
                    new_val = int(np.clip(
                        np.round(current + random.gauss(0, sigma)),
                        spec['min'],
                        spec['max']
                    ))
                    individual.genes[gene_name] = new_val
                
                elif spec['type'] in ['float', 'log_float']:
                    # Gaussian mutation
                    current = individual.genes[gene_name]
                    
                    if spec['type'] == 'log_float':
                        # Mutate in log space
                        log_current = np.log10(current)
                        log_min = np.log10(spec['min'])
                        log_max = np.log10(spec['max'])
                        sigma = (log_max - log_min) * 0.2
                        new_log = np.clip(log_current + random.gauss(0, sigma), log_min, log_max)
                        individual.genes[gene_name] = 10 ** new_log
                    else:
                        # Mutate in linear space
                        sigma = (spec['max'] - spec['min']) * 0.2
                        new_val = np.clip(
                            current + random.gauss(0, sigma),
                            spec['min'],
                            spec['max']
                        )
                        individual.genes[gene_name] = new_val
    
    def _evolve_generation(self, generation: int):
        """
        Evolve one generation
        
        Args:
            generation: Current generation number
        """
        # Sort population by fitness
        self.population.sort(key=lambda ind: ind.fitness if ind.fitness else -1.0, reverse=True)
        
        # Calculate statistics
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        avg_fitness = np.mean(fitnesses) if fitnesses else 0.0
        max_fitness = max(fitnesses) if fitnesses else 0.0
        min_fitness = min(fitnesses) if fitnesses else 0.0
        
        print(f"\n{'='*70}")
        print(f"Generation {generation}")
        print(f"{'='*70}")
        print(f"Best fitness: {max_fitness:.4f}")
        print(f"Avg fitness:  {avg_fitness:.4f}")
        print(f"Min fitness:  {min_fitness:.4f}")
        print(f"{'='*70}\n")
        
        # Record history
        self.history.append({
            'generation': generation,
            'best_fitness': float(max_fitness),
            'avg_fitness': float(avg_fitness),
            'min_fitness': float(min_fitness),
            'best_config': copy.deepcopy(self.population[0].genes) if self.population else None
        })
        
        # Elitism: preserve top individuals
        n_elite = max(1, int(self.population_size * self.elitism_rate))
        elite = self.population[:n_elite]
        
        # Generate new population
        new_population = elite.copy()
        
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            offspring1, offspring2 = self._crossover(parent1, parent2)
            
            # Mutation
            self._mutate(offspring1)
            self._mutate(offspring2)
            
            # Set generation
            offspring1.generation = generation
            offspring2.generation = generation
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        self.population = new_population[:self.population_size]
    
    def run_optimization(self):
        """Run genetic algorithm optimization"""
        print(f"\n{'='*70}")
        print(f"STARTING GENETIC ALGORITHM OPTIMIZATION FOR {self.model_name}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print(f"Mutation rate: {self.mutation_rate}")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"{'='*70}\n")
        
        # Initialize population
        self._initialize_population()
        
        # Evaluate initial population
        print("Evaluating initial population...")
        self._evaluate_population()
        
        # Evolution loop
        for generation in range(1, self.n_generations + 1):
            self._evolve_generation(generation)
            self._evaluate_population()
        
        # Final statistics
        print(f"\n{'='*70}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total evaluations: {self.evaluation_count}")
        print(f"Best fitness: {self.best_score:.4f}")
        print(f"Best generation: {self.best_individual.generation}")
        print(f"\nBest hyperparameters:")
        for key, value in self.best_individual.genes.items():
            if key not in ['model_name', 'config_name', 'loss', 'subset_size']:
                print(f"  {key:30s}: {value}")
        print(f"{'='*70}\n")
    
    def evaluate_best_model(self):
        """Evaluate best model on test set"""
        if self.best_individual is None:
            print("No best individual found. Run optimization first.")
            return None
        
        print(f"\n{'='*70}")
        print(f"EVALUATING BEST MODEL ON TEST SET")
        print(f"{'='*70}\n")
        
        config = self.best_individual.genes
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=config['batch_size'],
            shuffle=False
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
        
        # Save detailed results to JSON
        results_file = self.results_dir / f"{self.study_name}_{timestamp}_detailed.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'population_size': self.population_size,
                'n_generations': self.n_generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_rate': self.elitism_rate,
                'best_score': float(self.best_score),
                'best_config': self.best_individual.genes if self.best_individual else None,
                'best_generation': self.best_individual.generation if self.best_individual else None,
                'test_results': test_results,
                'history': self.history,
                'total_evaluations': self.evaluation_count
            }, f, indent=2)
        
        print(f"Detailed results saved to: {results_file}")
        
        # Save summary
        summary_file = self.results_dir / f"{self.study_name}_{timestamp}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"GENETIC ALGORITHM OPTIMIZATION RESULTS - {self.model_name}\n")
            f.write("="*80 + "\n\n")
            f.write(f"Optimization Study: {self.study_name}\n")
            f.write(f"Population Size: {self.population_size}\n")
            f.write(f"Generations: {self.n_generations}\n")
            f.write(f"Total Evaluations: {self.evaluation_count}\n")
            f.write(f"Mutation Rate: {self.mutation_rate}\n")
            f.write(f"Crossover Rate: {self.crossover_rate}\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("="*80 + "\n")
            f.write("*** BEST CONFIGURATION ***\n")
            f.write("="*80 + "\n")
            f.write(f"Best Validation F1 Score: {self.best_score:.4f}\n")
            f.write(f"Found in Generation: {self.best_individual.generation}\n\n")
            f.write("Hyperparameters:\n")
            for key, value in self.best_individual.genes.items():
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
            
            # Evolution progress
            f.write("="*80 + "\n")
            f.write("EVOLUTION PROGRESS\n")
            f.write("="*80 + "\n")
            for entry in self.history[-10:]:  # Last 10 generations
                f.write(f"\nGeneration {entry['generation']}:\n")
                f.write(f"  Best fitness: {entry['best_fitness']:.4f}\n")
                f.write(f"  Avg fitness:  {entry['avg_fitness']:.4f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Optimization completed successfully!\n")
            f.write("="*80 + "\n")
        
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Genetic Algorithm Optimization for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='GNN model to optimize')
    parser.add_argument('--population_size', type=int, default=20,
                        help='Population size (default: 20)')
    parser.add_argument('--n_generations', type=int, default=50,
                        help='Number of generations (default: 50)')
    parser.add_argument('--mutation_rate', type=float, default=0.2,
                        help='Mutation rate (default: 0.2)')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                        help='Crossover rate (default: 0.8)')
    parser.add_argument('--elitism_rate', type=float, default=0.1,
                        help='Elitism rate (default: 0.1)')
    parser.add_argument('--tournament_size', type=int, default=3,
                        help='Tournament size for selection (default: 3)')
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
    optimizer = GeneticOptimizer(
        model_name=args.model,
        population_size=args.population_size,
        n_generations=args.n_generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elitism_rate=args.elitism_rate,
        tournament_size=args.tournament_size,
        study_name=args.study_name,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        seed=args.seed
    )
    
    # Run optimization
    optimizer.run_optimization()
    
    # Evaluate on test set if requested
    test_results = None
    if args.evaluate_test:
        test_results = optimizer.evaluate_best_model()
    
    # Save results
    optimizer.save_results(test_results)
    
    print("\n" + "="*80)
    print("GENETIC ALGORITHM OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
