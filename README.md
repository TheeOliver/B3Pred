# BBB Permeability Predictor

Deep learning models for predicting Blood-Brain Barrier (BBB) permeability from molecular SMILES representations.

## Overview

This project implements Graph Neural Networks (GNNs) and traditional machine learning baselines for binary classification of BBB permeability. Models process molecular graphs derived from SMILES strings to predict whether compounds can cross the blood-brain barrier.

## Features

- **Multiple GNN Architectures**: GAT, GCN, GraphSAGE, GIN
- **Traditional ML Baselines**: Logistic Regression, Random Forest, SVM
- **Command-line Interface**: Easy training and evaluation with single commands
- **Experiment Tracking**: Automatic saving of models, configs, and results
- **Flexible Configuration**: JSON configs or command-line arguments

## Project Structure

```
bbb-graph-predictor/
├── configs/                      # Configuration files
│   ├── base_config.py           # Base project settings
│   ├── model_configs.py         # Model hyperparameters
│   └── experiment_configs/      # JSON experiment configs
├── src/                         # Source code
│   ├── data/                    # Data processing
│   │   └── featurizer.py       # SMILES to graph conversion
│   ├── models/                  # GNN models
│   │   ├── base.py             # Base GraphStack class
│   │   ├── gat.py              # Graph Attention Network
│   │   ├── gcn.py              # Graph Convolutional Network
│   │   ├── gin.py              # Graph Isomorphism Network
│   │   ├── graphsage.py        # GraphSAGE
│   │   └── predictor.py        # Full prediction model
│   ├── baselines/               # Traditional ML models
│   │   └── models.py
│   └── utils/                   # Training and evaluation utils
│       ├── train.py
│       └── evaluate.py
├── scripts/                     # Executable scripts
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Model evaluation script
│   └── run_baselines.py        # Baseline models script
├── data/                        # Dataset directory
│   ├── b3db_tanimoto_train.csv
│   ├── b3db_tanimoto_val.csv
│   └── b3db_tanimoto_test.csv
└── experiments/                 # Saved models and results

```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd bbb-graph-predictor

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training GNN Models

**1. Train with default settings:**
```bash
python scripts/train.py --model GAT --name gat_baseline
```

**2. Train with custom hyperparameters:**
```bash
python scripts/train.py --model GCN --name gcn_exp1 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size 64 \
    --graph_layers 4 \
    --graph_hidden_channels 128
```

**3. Train with config file:**
```bash
python scripts/train.py --config configs/experiment_configs/gat_baseline.json
```

**4. Train and evaluate on test set:**
```bash
python scripts/train.py --model GraphSAGE --name sage_baseline --test
```

### Evaluating Models

```bash
# Evaluate on test set
python scripts/evaluate.py --experiment gat_baseline --split test

# Evaluate on all splits
python scripts/evaluate.py --experiment gcn_exp1 --split all
```

### Training Baseline Models

```bash
# Train all baselines
python scripts/run_baselines.py --all

# Train specific baseline
python scripts/run_baselines.py --model random_forest
```

## Model Architectures

### Graph Neural Networks

- **GAT (Graph Attention Network)**: Uses multi-head attention mechanisms
- **GCN (Graph Convolutional Network)**: Classic spectral graph convolutions
- **GraphSAGE**: Sample and aggregate approach
- **GIN (Graph Isomorphism Network)**: Maximum expressive power

### Baselines

- **Logistic Regression**: Linear model with L2 regularization
- **Random Forest**: Ensemble of 300 decision trees
- **SVM**: RBF kernel with balanced class weights

All baseline models use Morgan fingerprints (2048 bits, radius 2).

## Configuration

### Available Hyperparameters

**Training:**
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 0.001)
- `weight_decay`: L2 regularization (default: 5e-4)
- `early_stopping_patience`: Patience for early stopping (default: 10)

**Graph Layers:**
- `graph_layers`: Number of graph convolution layers (default: 3)
- `graph_hidden_channels`: Hidden dimension (default: 64)
- `graph_dropout`: Dropout rate (default: 0.3)
- `use_graph_norm`: Use GraphNorm vs LayerNorm (default: true)

**Prediction Head:**
- `pred_layers`: Number of MLP layers (default: 3)
- `pred_hidden_channels`: Hidden dimension (default: 64)
- `pred_dropout`: Dropout rate (default: 0.3)

**GAT-Specific:**
- `attention_heads`: Number of attention heads (default: 4)
- `attention_dropout`: Attention dropout (default: 0.3)

## Data Format

CSV files with columns:
- `SMILES`: Molecular SMILES string
- `target`: Binary label (0 or 1) for BBB permeability

## Results

Results are automatically saved in `experiments/<experiment_name>/`:
- `config.json`: Experiment configuration
- `model.pth`: Trained model weights
- `results.json`: Evaluation metrics

### Metrics Reported

- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-score (weighted)
- AUC-ROC
- Confusion Matrix

## Advanced Usage

### Creating Custom Configs

Create a JSON file in `configs/experiment_configs/`:

```json
{
  "experiment_name": "my_experiment",
  "model_name": "GAT",
  "batch_size": 64,
  "epochs": 100,
  "learning_rate": 0.001,
  "graph_layers": 4,
  "graph_hidden_channels": 128,
  "attention_heads": 8
}
```

### GPU Training

```bash
python scripts/train.py --model GAT --name gat_gpu --device cuda
```

### Experiment Comparison

Compare multiple experiments by examining their `results.json` files:

```python
import json

with open('experiments/gat_baseline/results.json') as f:
    gat_results = json.load(f)

with open('experiments/gcn_baseline/results.json') as f:
    gcn_results = json.load(f)

print(f"GAT Test F1: {gat_results['test']['f1']:.4f}")
print(f"GCN Test F1: {gcn_results['test']['f1']:.4f}")
```

## Citation

If you use this code in your research, please cite:

```
[Your citation information]
```

## License

[Your license information]
