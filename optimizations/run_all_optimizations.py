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
import torch
import os


class BatchOptimizationRunner:
    """Run multiple optimization methods and compare results"""

    def __init__(self, methods=['bayesian', 'hyperband', 'cmaes', 'genetic', 'pso', 'abc', 'hillclimbing'],
                 models=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                 n_trials=100, evaluate_test=True, results_dir='./optimization_results',
                 subset_size=0.1, opt_epochs=10, top_k=10, full_epochs=100, auto_retrain=True):
        """
        Initialize batch runner

        Args:
            methods: List of optimization methods to run
            models: List of GNN models to optimize
            n_trials: Number of trials/iterations for each method
            evaluate_test: Whether to evaluate on test set
            results_dir: Directory to save all results
            subset_size: Fraction of training data to use during optimization
            opt_epochs: Number of epochs to use during optimization
            top_k: Number of top configurations to retrain on full data
            full_epochs: Number of epochs for full training
            auto_retrain: Whether to automatically retrain top-k models
        """
        self.methods = methods
        self.models = models
        self.n_trials = n_trials
        self.evaluate_test = evaluate_test
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization strategy parameters
        self.subset_size = subset_size
        self.opt_epochs = opt_epochs
        self.top_k = top_k
        self.full_epochs = full_epochs
        self.auto_retrain = auto_retrain

        self.results = []

    def run_optimization(self, method, model, seed=42):
        """
        Run a single optimization

        Args:
            method: Optimization method ('bayesian', 'hyperband', 'cmaes', etc.)
            model: GNN model to optimize
            seed: Random seed

        Returns:
            Return code from subprocess
        """
        study_name = f"{model}_{method}"

        print(f"\n{'=' * 80}")
        print(f"Running {method.upper()} optimization for {model}")
        print(f"{'=' * 80}\n")

        # Base command with common arguments
        base_cmd = [
            'python3', f'optimizations/{method}_optimization.py',
            '--model', model,
            '--study_name', study_name,
            '--results_dir', str(self.results_dir / method),
            '--seed', str(seed),
            '--subset_size', str(self.subset_size),
        ]
        
        # Method-specific arguments
        if method == 'bayesian':
            cmd = base_cmd + [
                '--n_trials', str(self.n_trials),
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'hyperband':
            cmd = base_cmd + [
                '--n_trials', str(self.n_trials),
                '--max_epochs', str(self.opt_epochs),  # Use opt_epochs as max_epochs
                '--reduction_factor', '3',
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'cmaes':
            n_iterations = max(30, self.n_trials // 2)  # CMA-ES typically needs fewer iterations
            cmd = base_cmd + [
                '--n_iterations', str(n_iterations),
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'genetic':
            cmd = base_cmd + [
                '--population_size', '10',
                '--n_generations', str(max(25, self.n_trials // 2)),
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'pso':
            cmd = base_cmd + [
                '--n_particles', '20',
                '--n_iterations', str(max(30, self.n_trials // 2)),
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'abc':
            cmd = base_cmd + [
                '--colony_size', '20',
                '--n_iterations', str(max(30, self.n_trials // 2)),
                '--limit', '10',  # Added missing limit parameter
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        elif method == 'hillclimbing':
            cmd = base_cmd + [
                '--n_iterations', str(max(20, self.n_trials // 3)),
                '--n_restarts', '5',
                '--opt_epochs', str(self.opt_epochs),
                '--top_k', str(self.top_k),
                '--full_epochs', str(self.full_epochs)
            ]
        else:
            raise ValueError(f"Unknown method: {method}")

        # Add evaluation flags
        if self.evaluate_test:
            cmd.append('--evaluate_test')
        if self.auto_retrain:
            cmd.append('--auto_retrain')

        # Run optimization
        try:
            print(f"Running command: {' '.join(cmd)}")
            # Set environment variables for CUDA
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            env['PYTHONUNBUFFERED'] = '1'
            
            result = subprocess.run(cmd, check=True, env=env)
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
        print(f"Optimization Data Subset: {self.subset_size*100:.1f}%")
        print(f"Optimization Epochs: {self.opt_epochs}")
        print(f"Top-K to Retrain: {self.top_k}")
        print(f"Full Training Epochs: {self.full_epochs}")
        print(f"Auto-Retrain: {self.auto_retrain}")
        print(f"Evaluate Test: {self.evaluate_test}")
        print(f"Results Directory: {self.results_dir}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
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
                        'opt_subset_size': data.get('opt_subset_size', 0.1),
                        'opt_epochs': data.get('opt_epochs', 10),
                    }

                    # Add test results if available
                    if data.get('test_results'):
                        test_res = data['test_results']
                        result_entry['test_accuracy'] = test_res.get('acc_target', None)
                        result_entry['test_f1'] = test_res.get('macro_f1', None)
                        result_entry['test_auc'] = test_res.get('auc_target', None)
                        result_entry['test_mcc'] = test_res.get('mcc_target', None)
                    
                    # Add final test results from retraining if available
                    if data.get('final_test_results'):
                        final_res = data['final_test_results']
                        result_entry['final_test_accuracy'] = final_res.get('acc_target', None)
                        result_entry['final_test_f1'] = final_res.get('macro_f1', None)
                        result_entry['final_test_auc'] = final_res.get('auc_target', None)
                        result_entry['final_test_mcc'] = final_res.get('mcc_target', None)

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
                    elif method == 'abc':
                        result_entry['colony_size'] = data.get('colony_size')
                        result_entry['n_iterations'] = data.get('n_iterations')
                        result_entry['limit'] = data.get('limit')
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
            f.write(f"Optimization Data Subset: {self.subset_size*100:.1f}%\n")
            f.write(f"Optimization Epochs: {self.opt_epochs}\n")
            f.write(f"Top-K Retrained: {self.top_k}\n")
            f.write(f"Full Training Epochs: {self.full_epochs}\n")
            f.write(f"Methods Compared: {', '.join(df['method'].unique())}\n")
            f.write(f"Models Optimized: {', '.join(df['model'].unique())}\n\n")

            # Overall best results
            f.write("=" * 80 + "\n")
            f.write("*** OVERALL BEST RESULTS ***\n")
            f.write("=" * 80 + "\n\n")

            if 'best_val_f1' in df.columns:
                if len(df) == 0 or df['best_val_f1'].isna().all():
                    print("No valid results found to compare.")
                    return                
                best_val_idx = df['best_val_f1'].idxmax()
                best_val = df.loc[best_val_idx]
                f.write("BEST VALIDATION F1 (during optimization):\n")
                f.write(f"  Model: {best_val['model']}\n")
                f.write(f"  Method: {best_val['method']}\n")
                f.write(f"  Validation F1: {best_val['best_val_f1']:.4f}\n\n")

            if 'final_test_f1' in df.columns and df['final_test_f1'].notna().any():
                best_test_idx = df['final_test_f1'].idxmax()
                best_test = df.loc[best_test_idx]
                f.write("BEST TEST F1 (after retraining on full data):\n")
                f.write(f"  Model: {best_test['model']}\n")
                f.write(f"  Method: {best_test['method']}\n")
                f.write(f"  Test F1: {best_test['final_test_f1']:.4f}\n")
                f.write(f"  Test Accuracy: {best_test['final_test_accuracy']:.4f}\n")
                if 'final_test_auc' in best_test and pd.notna(best_test['final_test_auc']):
                    f.write(f"  Test AUC: {best_test['final_test_auc']:.4f}\n")
                if 'final_test_mcc' in best_test and pd.notna(best_test['final_test_mcc']):
                    f.write(f"  Test MCC: {best_test['final_test_mcc']:.4f}\n")
                f.write("\n")
            elif 'test_f1' in df.columns and df['test_f1'].notna().any():
                best_test_idx = df['test_f1'].idxmax()
                best_test = df.loc[best_test_idx]
                f.write("BEST TEST F1 (direct evaluation):\n")
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

                    if 'final_test_f1' in row and pd.notna(row['final_test_f1']):
                        f.write(f"    Final Test F1: {row['final_test_f1']:.4f}\n")
                        f.write(f"    Final Test Accuracy: {row['final_test_accuracy']:.4f}\n")
                    elif 'test_f1' in row and pd.notna(row['test_f1']):
                        f.write(f"    Test F1: {row['test_f1']:.4f}\n")
                        f.write(f"    Test Accuracy: {row['test_accuracy']:.4f}\n")

                    if row['method'] == 'hyperband':
                        if 'completed_trials' in row and pd.notna(row['completed_trials']):
                            f.write(f"    Trials: {row['completed_trials']:.0f} completed, ")
                            f.write(f"{row['pruned_trials']:.0f} pruned\n")
                    elif row['method'] == 'cmaes' or row['method'] == 'abc':
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

                if 'final_test_f1' in method_df.columns and method_df['final_test_f1'].notna().any():
                    f.write(
                        f"  Average Final Test F1: {method_df['final_test_f1'].mean():.4f} ± {method_df['final_test_f1'].std():.4f}\n")
                elif 'test_f1' in method_df.columns and method_df['test_f1'].notna().any():
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

            if 'final_test_f1' in df.columns and df['final_test_f1'].notna().any():
                f.write("Final Test F1 Statistics:\n")
                test_f1_df = df[df['final_test_f1'].notna()]
                f.write(f"  Overall Mean: {test_f1_df['final_test_f1'].mean():.4f}\n")
                f.write(f"  Overall Std: {test_f1_df['final_test_f1'].std():.4f}\n")
                f.write(f"  Overall Min: {test_f1_df['final_test_f1'].min():.4f}\n")
                f.write(f"  Overall Max: {test_f1_df['final_test_f1'].max():.4f}\n\n")
            elif 'test_f1' in df.columns and df['test_f1'].notna().any():
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

            # Best overall based on final test if available, otherwise validation
            if 'final_test_f1' in df.columns and df['final_test_f1'].notna().any():
                best_overall = df.loc[df['final_test_f1'].idxmax()]
                f.write(f"1. BEST OVERALL CONFIGURATION (based on final test F1):\n")
                f.write(f"   Use {best_overall['model']} optimized with {best_overall['method'].upper()}\n")
                f.write(f"   Expected Test F1: {best_overall['final_test_f1']:.4f}\n")
                f.write(f"   Test Accuracy: {best_overall['final_test_accuracy']:.4f}\n\n")
            else:
                best_overall = df.loc[df['best_val_f1'].idxmax()]
                f.write(f"1. BEST OVERALL CONFIGURATION (based on validation F1):\n")
                f.write(f"   Use {best_overall['model']} optimized with {best_overall['method'].upper()}\n")
                f.write(f"   Expected Validation F1: {best_overall['best_val_f1']:.4f}\n\n")

            # Method recommendation
            if 'final_test_f1' in df.columns and df['final_test_f1'].notna().any():
                method_avg = df.groupby('method')['final_test_f1'].agg(['mean', 'std']).dropna()
                if not method_avg.empty:
                    best_method = method_avg['mean'].idxmax()
                    f.write(f"2. MOST RELIABLE METHOD (based on final test F1):\n")
                    f.write(f"   {best_method.upper()} achieved the best average results\n")
                    f.write(
                        f"   Average Final Test F1: {method_avg.loc[best_method, 'mean']:.4f} ± {method_avg.loc[best_method, 'std']:.4f}\n\n")
            else:
                method_avg = df.groupby('method')['best_val_f1'].agg(['mean', 'std'])
                best_method = method_avg['mean'].idxmax()
                f.write(f"2. MOST RELIABLE METHOD (based on validation F1):\n")
                f.write(f"   {best_method.upper()} achieved the best average results\n")
                f.write(
                    f"   Average Validation F1: {method_avg.loc[best_method, 'mean']:.4f} ± {method_avg.loc[best_method, 'std']:.4f}\n\n")

            # Model recommendation
            if 'final_test_f1' in df.columns and df['final_test_f1'].notna().any():
                model_avg = df.groupby('model')['final_test_f1'].agg(['mean', 'std']).dropna()
                if not model_avg.empty:
                    best_model = model_avg['mean'].idxmax()
                    f.write(f"3. BEST MODEL ARCHITECTURE (based on final test F1):\n")
                    f.write(f"   {best_model} performed best on average\n")
                    f.write(
                        f"   Average Final Test F1: {model_avg.loc[best_model, 'mean']:.4f} ± {model_avg.loc[best_model, 'std']:.4f}\n\n")
            else:
                model_avg = df.groupby('model')['best_val_f1'].agg(['mean', 'std'])
                best_model = model_avg['mean'].idxmax()
                f.write(f"3. BEST MODEL ARCHITECTURE (based on validation F1):\n")
                f.write(f"   {best_model} performed best on average\n")
                f.write(
                    f"   Average Validation F1: {model_avg.loc[best_model, 'mean']:.4f} ± {model_avg.loc[best_model, 'std']:.4f}\n\n")

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
  # Run all methods on all models with default settings
  python run_all_optimizations.py --n_trials 100

  # Run with custom optimization settings
  python run_all_optimizations.py --subset_size 0.2 --opt_epochs 15 --top_k 15 --full_epochs 150

  # Run only Bayesian optimization
  python run_all_optimizations.py --methods bayesian --n_trials 150

  # Run evolutionary methods only
  python run_all_optimizations.py --methods genetic pso abc --n_trials 100

  # Run on specific models only
  python run_all_optimizations.py --models GAT GCN --n_trials 100

  # Quick test run
  python run_all_optimizations.py --n_trials 10 --no_evaluate_test --no_auto_retrain

  # Only generate comparison report from existing results
  python run_all_optimizations.py --compare_only --results_dir ./optimization_results
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
    
    # New arguments for optimization strategy
    parser.add_argument('--subset_size', type=float, default=0.1,
                        help='Fraction of training data to use during optimization (default: 0.1)')
    parser.add_argument('--opt_epochs', type=int, default=10,
                        help='Number of epochs to use during optimization (default: 10)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top configurations to retrain on full data (default: 10)')
    parser.add_argument('--full_epochs', type=int, default=100,
                        help='Number of epochs for full training (default: 100)')
    parser.add_argument('--no_auto_retrain', action='store_true',
                        help='Disable automatic retraining of top-k models')

    args = parser.parse_args()

    # Print CUDA info at start
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
    print("")

    # Create runner
    runner = BatchOptimizationRunner(
        methods=args.methods,
        models=args.models,
        n_trials=args.n_trials,
        evaluate_test=not args.no_evaluate_test,
        results_dir=args.results_dir,
        subset_size=args.subset_size,
        opt_epochs=args.opt_epochs,
        top_k=args.top_k,
        full_epochs=args.full_epochs,
        auto_retrain=not args.no_auto_retrain
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