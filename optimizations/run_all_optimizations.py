#!/usr/bin/env python3
"""
Batch Optimization Runner
Runs multiple optimization methods on all models and generates comparison report
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class BatchOptimizationRunner:
    """Run multiple optimization methods and compare results"""

    def __init__(self, methods=['bayesian', 'hyperband', 'cmaes', 'genetic', 'pso', 'abc', 'hillclimbing'],
                 models=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                 n_trials=100, evaluate_test=True, results_dir='./optimization_results'):
        """
        Initialize batch runner

        Args:
            methods: List of optimization methods to run
            models: List of GNN models to optimize
            n_trials: Number of trials/iterations for each method
            evaluate_test: Whether to evaluate on test set
            results_dir: Directory to save all results
        """
        self.methods = methods
        self.models = models
        self.n_trials = n_trials
        self.evaluate_test = evaluate_test
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.results = []

    def run_optimization(self, method, model, seed=42):
        """
        Run a single optimization

        Args:
            method: Optimization method ('bayesian', 'hyperband', or 'cmaes')
            model: GNN model to optimize
            seed: Random seed

        Returns:
            Return code from subprocess
        """
        study_name = f"{model}_{method}"

        print(f"\n{'=' * 80}")
        print(f"Running {method.upper()} optimization for {model}")
        print(f"{'=' * 80}\n")

        # Build command
        if method == 'bayesian':
            cmd = [
                'python', 'optimizations/bayesian_optimization.py',
                '--model', model,
                '--n_trials', str(self.n_trials),
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'hyperband':
            cmd = [
                'python', 'optimizations/hyperband_optimization.py',
                '--model', model,
                '--n_trials', str(self.n_trials),
                '--max_epochs', '81',
                '--reduction_factor', '3',
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'cmaes':
            n_iterations = max(30, self.n_trials // 2)  # CMA-ES typically needs fewer iterations
            cmd = [
                'python', 'optimizations/cmaes_optimization.py',
                '--model', model,
                '--n_iterations', str(n_iterations),
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'genetic':
            cmd = [
                'python', 'optimizations/genetic_optimization.py',
                '--model', model,
                '--population_size', '20',
                '--n_generations', str(max(30, self.n_trials // 2)),
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'pso':
            cmd = [
                'python', 'optimizations/pso_optimization.py',
                '--model', model,
                '--n_particles', '20',
                '--n_iterations', str(max(30, self.n_trials // 2)),
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'abc':
            cmd = [
                'python', 'optimizations/abc_optimization.py',
                '--model', model,
                '--colony_size', '20',
                '--n_iterations', str(max(30, self.n_trials // 2)),
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        elif method == 'hillclimbing':
            cmd = [
                'python', 'optimizations/hillclimbing_optimization.py',
                '--model', model,
                '--n_iterations', str(max(20, self.n_trials // 3)),
                '--n_restarts', '5',
                '--study_name', study_name,
                '--results_dir', str(self.results_dir / method),
                '--seed', str(seed)
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

        if self.evaluate_test:
            cmd.append('--evaluate_test')

        # Run optimization
        try:
            result = subprocess.run(cmd, check=True)
            return result.returncode
        except subprocess.CalledProcessError as e:
            print(f"Error running {method} for {model}: {e}")
            return e.returncode

    def run_all(self):
        """Run all optimization methods for all models"""
        print(f"\n{'=' * 80}")
        print(f"BATCH OPTIMIZATION RUNNER")
        print(f"{'=' * 80}")
        print(f"Methods: {', '.join(self.methods)}")
        print(f"Models: {', '.join(self.models)}")
        print(f"Trials/Iterations: {self.n_trials}")
        print(f"Evaluate Test: {self.evaluate_test}")
        print(f"Results Directory: {self.results_dir}")
        print(f"{'=' * 80}\n")

        total_runs = len(self.methods) * len(self.models)
        current_run = 0

        for method in self.methods:
            for model in self.models:
                current_run += 1
                print(f"\nProgress: {current_run}/{total_runs}")
                self.run_optimization(method, model)

        print(f"\n{'=' * 80}")
        print(f"ALL OPTIMIZATIONS COMPLETE!")
        print(f"{'=' * 80}\n")

    def collect_results(self):
        """Collect results from all optimization runs"""
        print("Collecting results from all optimization runs...")

        results = []

        for method in self.methods:
            method_dir = self.results_dir / method
            if not method_dir.exists():
                continue

            # Find all detailed JSON files
            for json_file in method_dir.glob("*_detailed.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    result_entry = {
                        'method': method,
                        'model': data['model_name'],
                        'study_name': json_file.stem.replace('_detailed', ''),
                        'best_val_f1': data['best_score'],
                    }

                    # Add test results if available
                    if data.get('test_results'):
                        test_res = data['test_results']
                        result_entry['test_accuracy'] = test_res.get(f"acc_{data['model_name']}", None)
                        result_entry['test_f1'] = test_res.get('macro_f1', None)
                        result_entry['test_auc'] = test_res.get(f"auc_{data['model_name']}", None)
                        result_entry['test_mcc'] = test_res.get(f"mcc_{data['model_name']}", None)

                    # Add method-specific info
                    if method == 'bayesian':
                        result_entry['n_trials'] = data.get('n_trials')
                    elif method == 'hyperband':
                        result_entry['n_trials'] = data.get('n_trials')
                        result_entry['completed_trials'] = data.get('completed_trials')
                        result_entry['pruned_trials'] = data.get('pruned_trials')
                    elif method == 'cmaes':
                        result_entry['n_iterations'] = data.get('n_iterations')
                        result_entry['total_evaluations'] = data.get('total_evaluations')

                    results.append(result_entry)

                except Exception as e:
                    print(f"Error reading {json_file}: {e}")

        return pd.DataFrame(results)

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        df = self.collect_results()

        if df.empty:
            print("No results found to compare!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"comparison_report_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("HYPERPARAMETER OPTIMIZATION COMPARISON REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Optimization Runs: {len(df)}\n")
            f.write(f"Methods Compared: {', '.join(df['method'].unique())}\n")
            f.write(f"Models Optimized: {', '.join(df['model'].unique())}\n\n")

            # Overall best results
            f.write("=" * 80 + "\n")
            f.write("*** OVERALL BEST RESULTS ***\n")
            f.write("=" * 80 + "\n\n")

            if 'best_val_f1' in df.columns:
                best_val = df.loc[df['best_val_f1'].idxmax()]
                f.write("BEST VALIDATION F1:\n")
                f.write(f"  Model: {best_val['model']}\n")
                f.write(f"  Method: {best_val['method']}\n")
                f.write(f"  Validation F1: {best_val['best_val_f1']:.4f}\n\n")

            if 'test_f1' in df.columns and df['test_f1'].notna().any():
                best_test = df.loc[df['test_f1'].idxmax()]
                f.write("BEST TEST F1:\n")
                f.write(f"  Model: {best_test['model']}\n")
                f.write(f"  Method: {best_test['method']}\n")
                f.write(f"  Test F1: {best_test['test_f1']:.4f}\n")
                f.write(f"  Test Accuracy: {best_test['test_accuracy']:.4f}\n")
                if 'test_auc' in best_test and pd.notna(best_test['test_auc']):
                    f.write(f"  Test AUC: {best_test['test_auc']:.4f}\n")
                if 'test_mcc' in best_test and pd.notna(best_test['test_mcc']):
                    f.write(f"  Test MCC: {best_test['test_mcc']:.4f}\n")
                f.write("\n")

            # Results by model
            f.write("=" * 80 + "\n")
            f.write("RESULTS BY MODEL\n")
            f.write("=" * 80 + "\n\n")

            for model in sorted(df['model'].unique()):
                f.write(f"\n{model}:\n")
                f.write("-" * 40 + "\n")

                model_df = df[df['model'] == model].sort_values('best_val_f1', ascending=False)

                for _, row in model_df.iterrows():
                    f.write(f"\n  Method: {row['method'].upper()}\n")
                    f.write(f"    Validation F1: {row['best_val_f1']:.4f}\n")

                    if 'test_f1' in row and pd.notna(row['test_f1']):
                        f.write(f"    Test F1: {row['test_f1']:.4f}\n")
                        f.write(f"    Test Accuracy: {row['test_accuracy']:.4f}\n")

                    if row['method'] == 'hyperband':
                        if 'completed_trials' in row and pd.notna(row['completed_trials']):
                            f.write(f"    Trials: {row['completed_trials']:.0f} completed, ")
                            f.write(f"{row['pruned_trials']:.0f} pruned\n")
                    elif row['method'] == 'cmaes':
                        if 'total_evaluations' in row and pd.notna(row['total_evaluations']):
                            f.write(f"    Total Evaluations: {row['total_evaluations']:.0f}\n")

            # Results by method
            f.write("\n" + "=" * 80 + "\n")
            f.write("RESULTS BY METHOD\n")
            f.write("=" * 80 + "\n\n")

            for method in sorted(df['method'].unique()):
                f.write(f"\n{method.upper()}:\n")
                f.write("-" * 40 + "\n")

                method_df = df[df['method'] == method]

                f.write(
                    f"  Average Validation F1: {method_df['best_val_f1'].mean():.4f} ± {method_df['best_val_f1'].std():.4f}\n")

                if 'test_f1' in method_df.columns and method_df['test_f1'].notna().any():
                    f.write(
                        f"  Average Test F1: {method_df['test_f1'].mean():.4f} ± {method_df['test_f1'].std():.4f}\n")

                f.write(f"  Best Model: {method_df.loc[method_df['best_val_f1'].idxmax()]['model']}\n")
                f.write(f"  Best Validation F1: {method_df['best_val_f1'].max():.4f}\n")

            # Statistical summary
            f.write("\n" + "=" * 80 + "\n")
            f.write("STATISTICAL SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            if 'best_val_f1' in df.columns:
                f.write("Validation F1 Statistics:\n")
                f.write(f"  Overall Mean: {df['best_val_f1'].mean():.4f}\n")
                f.write(f"  Overall Std: {df['best_val_f1'].std():.4f}\n")
                f.write(f"  Overall Min: {df['best_val_f1'].min():.4f}\n")
                f.write(f"  Overall Max: {df['best_val_f1'].max():.4f}\n\n")

            if 'test_f1' in df.columns and df['test_f1'].notna().any():
                f.write("Test F1 Statistics:\n")
                test_f1_df = df[df['test_f1'].notna()]
                f.write(f"  Overall Mean: {test_f1_df['test_f1'].mean():.4f}\n")
                f.write(f"  Overall Std: {test_f1_df['test_f1'].std():.4f}\n")
                f.write(f"  Overall Min: {test_f1_df['test_f1'].min():.4f}\n")
                f.write(f"  Overall Max: {test_f1_df['test_f1'].max():.4f}\n\n")

            # Recommendations
            f.write("=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n\n")

            best_overall = df.loc[df['best_val_f1'].idxmax()]
            f.write(f"1. BEST OVERALL CONFIGURATION:\n")
            f.write(f"   Use {best_overall['model']} optimized with {best_overall['method'].upper()}\n")
            f.write(f"   Expected Validation F1: {best_overall['best_val_f1']:.4f}\n\n")

            # Method recommendation
            method_avg = df.groupby('method')['best_val_f1'].agg(['mean', 'std'])
            best_method = method_avg['mean'].idxmax()
            f.write(f"2. MOST RELIABLE METHOD:\n")
            f.write(f"   {best_method.upper()} achieved the best average results\n")
            f.write(
                f"   Average F1: {method_avg.loc[best_method, 'mean']:.4f} ± {method_avg.loc[best_method, 'std']:.4f}\n\n")

            # Model recommendation
            model_avg = df.groupby('model')['best_val_f1'].agg(['mean', 'std'])
            best_model = model_avg['mean'].idxmax()
            f.write(f"3. BEST MODEL ARCHITECTURE:\n")
            f.write(f"   {best_model} performed best on average\n")
            f.write(
                f"   Average F1: {model_avg.loc[best_model, 'mean']:.4f} ± {model_avg.loc[best_model, 'std']:.4f}\n\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        print(f"\nComparison report saved to: {report_file}")

        # Also save as CSV for easy analysis
        csv_file = self.results_dir / f"comparison_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Results CSV saved to: {csv_file}")

        return df


def main():
    parser = argparse.ArgumentParser(
        description="Batch Optimization Runner for BBB Prediction Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all methods on all models
  python run_all_optimizations.py --n_trials 100

  # Run only Bayesian optimization
  python run_all_optimizations.py --methods bayesian --n_trials 150

  # Run evolutionary methods only
  python run_all_optimizations.py --methods genetic pso abc --n_trials 100

  # Run on specific models only
  python run_all_optimizations.py --models GAT GCN --n_trials 100

  # Quick test run
  python run_all_optimizations.py --n_trials 10 --no_evaluate_test
        """
    )

    parser.add_argument('--methods', nargs='+',
                        choices=['bayesian', 'hyperband', 'cmaes', 'genetic', 'pso', 'abc', 'hillclimbing'],
                        default=['bayesian', 'hyperband', 'cmaes', 'genetic', 'pso', 'abc', 'hillclimbing'],
                        help='Optimization methods to run')
    parser.add_argument('--models', nargs='+',
                        choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        default=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                        help='Models to optimize')
    parser.add_argument('--n_trials', type=int, default=100,
                        help='Number of trials/iterations (default: 100)')
    parser.add_argument('--no_evaluate_test', action='store_true',
                        help='Skip test set evaluation')
    parser.add_argument('--results_dir', type=str, default='./optimization_results',
                        help='Directory to save results')
    parser.add_argument('--compare_only', action='store_true',
                        help='Only generate comparison report from existing results')

    args = parser.parse_args()

    # Create runner
    runner = BatchOptimizationRunner(
        methods=args.methods,
        models=args.models,
        n_trials=args.n_trials,
        evaluate_test=not args.no_evaluate_test,
        results_dir=args.results_dir
    )

    # Run optimizations or just compare
    if not args.compare_only:
        runner.run_all()

    # Generate comparison report
    print("\nGenerating comparison report...")
    runner.generate_comparison_report()

    print("\n" + "=" * 80)
    print("BATCH OPTIMIZATION COMPLETE!")
    print(f"All results saved to: {args.results_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()