# Hyperparameter Optimization for BBB Prediction Models

This directory contains three state-of-the-art hyperparameter optimization methods for Graph Neural Networks (GNNs) used in Blood-Brain Barrier (BBB) permeability prediction.

## Available Optimization Methods

### 1. Bayesian Optimization (`bayesian_optimization.py`)
Uses Tree-structured Parzen Estimator (TPE) to intelligently explore the hyperparameter space.

**Advantages:**
- Efficient exploration of hyperparameter space
- Good for expensive objective functions
- Provides uncertainty estimates
- Works well with 100-200 trials

**Best for:** When you want high-quality results with moderate computational budget

### 2. Hyperband Optimization (`hyperband_optimization.py`)
Uses successive halving to efficiently allocate resources to promising configurations.

**Advantages:**
- Automatically prunes poor configurations early
- Very efficient use of computational resources
- Can handle large numbers of trials
- Saves ~40-60% of training time compared to grid search

**Best for:** When you want to explore many configurations quickly

### 3. CMA-ES Optimization (`cmaes_optimization.py`)
Covariance Matrix Adaptation Evolution Strategy - a powerful evolutionary algorithm.

**Advantages:**
- Excellent for continuous optimization
- Adapts search distribution during optimization
- Robust to noisy objectives
- Good convergence properties

**Best for:** When you want robust optimization with good theoretical guarantees

---

## Installation

### Required Dependencies

```bash
# Core dependencies (already in your environment)
pip install torch torch-geometric pandas numpy scikit-learn

# Optimization-specific dependencies
pip install optuna  # For Bayesian and Hyperband
pip install cma     # For CMA-ES

# Optional: For better visualization and logging
pip install wandb plotly
```

---

## Usage Examples

### Bayesian Optimization

#### Basic usage:
```bash
python bayesian_optimization.py --model GAT --n_trials 100
```

#### With test evaluation:
```bash
python bayesian_optimization.py --model GCN --n_trials 150 --evaluate_test
```

#### Custom study name and results directory:
```bash
python bayesian_optimization.py \
    --model GINE \
    --n_trials 200 \
    --study_name "gine_final_run" \
    --results_dir "./my_results" \
    --evaluate_test
```

### Hyperband Optimization

#### Basic usage:
```bash
python hyperband_optimization.py --model GAT --n_trials 100
```

#### Custom max epochs and reduction factor:
```bash
python hyperband_optimization.py \
    --model GraphSAGE \
    --n_trials 150 \
    --max_epochs 81 \
    --reduction_factor 3 \
    --evaluate_test
```

#### Fast exploration with many trials:
```bash
python hyperband_optimization.py \
    --model GIN \
    --n_trials 300 \
    --max_epochs 27 \
    --reduction_factor 3
```

### CMA-ES Optimization

#### Basic usage:
```bash
python cmaes_optimization.py --model GAT --n_iterations 50
```

#### With custom population size:
```bash
python cmaes_optimization.py \
    --model GCN \
    --n_iterations 100 \
    --population_size 20 \
    --evaluate_test
```

#### Quick optimization:
```bash
python cmaes_optimization.py \
    --model GINE \
    --n_iterations 30 \
    --population_size 12
```

---

## Command-Line Arguments

### Common Arguments (all methods)

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | GNN model: GAT, GCN, GraphSAGE, GIN, or GINE | **Required** |
| `--study_name` | Name for the optimization study | `{model}_[method]_opt` |
| `--results_dir` | Directory to save results | `experiments/[method]_optimization/` |
| `--seed` | Random seed for reproducibility | `42` |
| `--evaluate_test` | Evaluate best model on test set | `False` |

### Bayesian Optimization Specific

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_trials` | Number of optimization trials | `100` |

### Hyperband Optimization Specific

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_trials` | Number of optimization trials | `100` |
| `--max_epochs` | Maximum epochs for training | `81` |
| `--reduction_factor` | Hyperband reduction factor | `3` |

### CMA-ES Optimization Specific

| Argument | Description | Default |
|----------|-------------|---------|
| `--n_iterations` | Number of CMA-ES generations | `50` |
| `--population_size` | Population size per generation | `auto` |

---

## Optimized Hyperparameters

All methods optimize the following hyperparameters:

### Graph Model Parameters
- `graph_layers`: Number of graph convolutional layers (2-5)
- `graph_hidden_channels`: Hidden dimension (32, 64, 128, 256, 512)
- `graph_dropouts`: Dropout rate in graph layers (0.0-0.6)
- `graph_norm`: Use graph normalization (True/False)

### Predictor Parameters
- `pred_layers`: Number of predictor layers (2-4)
- `pred_hidden_channels`: Predictor hidden dimension (32, 64, 128, 256)
- `pred_dropouts`: Dropout rate in predictor (0.0-0.6)

### Training Parameters
- `batch_size`: Batch size (16, 32, 64, 128)
- `lr`: Learning rate (1e-5 to 1e-2, log scale)
- `epochs`: Number of training epochs (20-100)

### GAT-Specific Parameters
- `attention_heads`: Number of attention heads (2, 4, 8)
- `attention_dropouts`: Attention dropout rate (0.0-0.6)

---

## Output Files

Each optimization run generates three files:

### 1. Detailed Results (JSON)
`{study_name}_{timestamp}_detailed.json`

Contains:
- Complete configuration for all trials/evaluations
- Validation metrics for each trial
- Test set results (if `--evaluate_test` used)
- Full hyperparameter history

### 2. Summary Results (TXT)
`{study_name}_{timestamp}_summary.txt`

Contains:
- **Highlighted best configuration**
- **Test set performance** (if evaluated)
- Top 10 trials/evaluations
- Optimization statistics
- **Easy-to-read format for quick review**

### 3. Optimization Object (PKL)
`{study_name}_{timestamp}_[study/cmaes].pkl`

Contains:
- Complete optimization state
- Can be loaded for further analysis
- Useful for resuming optimization

---

## Recommended Workflows

### Quick Exploration (1-2 hours)
```bash
# Use Hyperband for fast exploration
python hyperband_optimization.py --model GAT --n_trials 100 --max_epochs 27
python hyperband_optimization.py --model GCN --n_trials 100 --max_epochs 27
python hyperband_optimization.py --model GINE --n_trials 100 --max_epochs 27
```

### Thorough Optimization (4-8 hours)
```bash
# Use Bayesian Optimization
python bayesian_optimization.py --model GAT --n_trials 150 --evaluate_test
python bayesian_optimization.py --model GCN --n_trials 150 --evaluate_test
python bayesian_optimization.py --model GINE --n_trials 150 --evaluate_test
```

### Deep Optimization (8-16 hours)
```bash
# Use CMA-ES for robust optimization
python cmaes_optimization.py --model GAT --n_iterations 100 --evaluate_test
python cmaes_optimization.py --model GCN --n_iterations 100 --evaluate_test
python cmaes_optimization.py --model GINE --n_iterations 100 --evaluate_test
```

### Comprehensive Comparison
Run all three methods on the same model and compare results:

```bash
# All three methods on GAT
python bayesian_optimization.py --model GAT --n_trials 100 --study_name "gat_bayesian" --evaluate_test
python hyperband_optimization.py --model GAT --n_trials 100 --study_name "gat_hyperband" --evaluate_test
python cmaes_optimization.py --model GAT --n_iterations 50 --study_name "gat_cmaes" --evaluate_test
```

---

## Tips and Best Practices

### 1. Start Small
Begin with a small number of trials/iterations to ensure everything works:
```bash
python bayesian_optimization.py --model GAT --n_trials 10
```

### 2. Use Test Evaluation Sparingly
Only use `--evaluate_test` for final runs to avoid overfitting to the test set.

### 3. Monitor Progress
All scripts print progress updates. Check these to ensure optimization is working correctly.

### 4. Compare Methods
Different methods may find different optima. Run multiple methods for important experiments.

### 5. Reproducibility
Always set a fixed seed for reproducible results:
```bash
python bayesian_optimization.py --model GAT --n_trials 100 --seed 42
```

### 6. Resource Allocation

**For limited compute:**
- Use Hyperband with higher `n_trials` (200-300)
- Use smaller `max_epochs` (27 or 9)

**For moderate compute:**
- Use Bayesian Optimization with 100-150 trials

**For extensive compute:**
- Use CMA-ES with 100+ iterations
- Use Bayesian Optimization with 200+ trials

---

## Interpreting Results

### Key Metrics to Focus On

1. **Validation F1 Score**: Primary optimization metric
2. **Test Accuracy**: Final model performance
3. **Test AUC**: Model discrimination ability
4. **Test MCC**: Matthews Correlation Coefficient (accounts for imbalance)

### Reading the Summary File

The summary file highlights:
- **Best configuration**: Top section, clearly marked
- **Test performance**: Shown if `--evaluate_test` was used
- **Top 10 trials**: Shows configuration diversity
- **Method-specific statistics**: Efficiency metrics

### Comparing Across Models

After running optimization for all models, compare:
1. Best validation F1 scores
2. Test set performance (if evaluated)
3. Configuration stability (do top 10 configs have similar hyperparameters?)
4. Training time and efficiency

---

## Troubleshooting

### Out of Memory Errors
- Reduce `batch_size` in the parameter space
- Use smaller `graph_hidden_channels`
- Use fewer `graph_layers`

### Slow Training
- Reduce `max_epochs` (for Hyperband)
- Reduce `n_trials` or `n_iterations`
- Use smaller dataset subset initially

### Poor Results
- Increase number of trials/iterations
- Check if data is properly normalized
- Verify train/val/test splits are balanced
- Try different random seeds

### Optimization Not Converging (CMA-ES)
- Increase `n_iterations`
- Adjust `population_size`
- Check if objective function is too noisy

---

## Example: Complete Optimization Pipeline

```bash
#!/bin/bash

# Optimize all models with Bayesian Optimization
for model in GAT GCN GraphSAGE GIN GINE; do
    echo "Optimizing $model with Bayesian Optimization..."
    python bayesian_optimization.py \
        --model $model \
        --n_trials 150 \
        --study_name "${model}_bayesian" \
        --evaluate_test \
        --seed 42
done

# Compare with Hyperband
for model in GAT GCN GraphSAGE GIN GINE; do
    echo "Optimizing $model with Hyperband..."
    python hyperband_optimization.py \
        --model $model \
        --n_trials 200 \
        --max_epochs 81 \
        --study_name "${model}_hyperband" \
        --evaluate_test \
        --seed 42
done

echo "Optimization complete! Check results in experiments/ directory"
```

---

## Citation

If you use these optimization methods in your research, please cite the relevant papers:

**Bayesian Optimization (TPE):**
```
Bergstra, J., Yamins, D., Cox, D. D. (2013). 
Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures.
ICML 2013.
```

**Hyperband:**
```
Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2018).
Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.
JMLR 2018.
```

**CMA-ES:**
```
Hansen, N., Ostermeier, A. (2001).
Completely Derandomized Self-Adaptation in Evolution Strategies.
Evolutionary Computation, 9(2), 159-195.
```

---

## Contact and Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the example commands
3. Examine the detailed JSON output for debugging

Good luck with your optimization!
