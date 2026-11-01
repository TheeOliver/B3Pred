# B3Pred: BBB permeability experiments

This repository contains the code and data used for the empirical benchmark in **“Blood–Brain Barrier Permeability Prediction with Graph Neural Networks: A Systematic Review and Benchmark.”** It brings the in-house graph models and locally available comparison implementations into one experiment workspace. Read [`docs/reproducibility.md`](docs/reproducibility.md) before reproducing paper numbers; some literature baselines were evaluated via a remote service or fixed checkpoints and cannot be retrained from this repository. The root MIT license applies to the B3Pred project code; external baseline code and bundled third-party assets remain subject to their own notices and terms. Some copied baseline snapshots have no license file and require permission checks before public redistribution.

## Quick start

The examples below run the in-house GNN pipeline. Run commands from the repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build that matches your CPU/CUDA setup, then install the compatible PyTorch Geometric package/wheels as described by the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). Finish with:

```bash
python -m pip install -r requirements.txt
```

The default dependency list is for the core GNN and classical fingerprint baselines; each external model has separate requirements in `baselines/`.

Train and test GINE using the checked-in Tanimoto split:

```bash
python experiments.py --model GINE --name gine_first_run --train --test --save --seed 42
```

The run writes a checkpoint, configuration, and test metrics under `experiments/gine_first_run/`. The location can be changed with `B3PRED_EXPERIMENTS_ROOT`; the input data directory can be changed with `B3PRED_DATA`.

Train a different architecture by changing `--model` to `GAT`, `GCN`, `GraphSAGE`, or `GIN`. The article's four-model comparison is GAT, GCN, GraphSAGE, and GINE; GIN is present in the code and in some optimization runs but is not one of the four headline architectures.

## What is here

| Path | Contents |
| --- | --- |
| `configs/`, `graph/`, `model/`, `scripts/` | In-house graph featurization, GAT/GCN/GraphSAGE/GIN/GINE architectures, training, and metrics. |
| `experiments.py` | Train, test, and simple grid-search command line interface. |
| `optimizations/` | Bayesian, Hyperband, CMA-ES, genetic, PSO, ABC, hill climbing, grid, random, tabu, and simulated annealing HPO. |
| `classical_ml.py` | In-house Morgan fingerprint LR, RF, and XGBoost baselines. |
| `baselines/` | Source for external comparison implementations, organized by model family with retained upstream notices. |
| `data/` | Fixed B3DB and MoleculeNet Tanimoto splits and the external LCD/S-data evaluation set. |
| `results/` | Historical optimization outputs, selected checkpoints, and compact external-baseline metrics. New runs go to ignored `experiments/` by default. |
| `docs/reproducibility.md` | Dataset definitions, model inventory, reproduction steps, result provenance, and known constraints. |
| `paper/` | Manuscript and bibliography source plus figures available in the workspace. |
| `baselines/README.md` | Baseline-specific commands and environment notes. |
| `configs/experiment_template.json` | Copyable JSON configuration for a named GNN run. |
| `scripts/summarize_results.py` | Convert run metric JSON files to a CSV for review. |

## Common workflows

### Configure and train

Copy `configs/experiment_template.json`, give it a unique `config_name`, and edit the model and hyperparameters. Then run:

```bash
python experiments.py --model GINE --name my_gine --config_path configs/experiment_template.json --train --test --save --seed 42
```

The CLI's `--model` and `--name` identify the selected architecture and output folder; keep `config_name` in the JSON equal to `--name`. See `python experiments.py --help` for all options.

### Test a saved model on another CSV

```bash
python experiments.py --model GINE --name my_gine --test --test_data data/prepared/sdata_prepared.csv --test_target_col target --results_name sdata_metrics
```

### Run HPO

Start with one model and one method. Search runs can be expensive; increase the budget after checking a short run.

```bash
python optimizations/run_all_optimizations.py \
  --methods bayesian \
  --models GINE \
  --n_trials 10 \
  --subset_size 0.1 \
  --opt_epochs 10 \
  --no_auto_retrain \
  --results_dir experiments/hpo_smoke
```

For a longer run, remove `--no_auto_retrain`, raise `--n_trials`, and set `--top_k` and `--full_epochs` explicitly. To run the paper's broader sweep, provide the desired methods/models and the study's original compute budget. HPO outputs include configurations, detailed metrics, and checkpoints; output directories are intentionally not committed by default.

### Run the Morgan fingerprint baselines

```bash
python classical_ml.py --help
```

This script expects the checked-in train/validation/test split files and S-data. See the help output and [`docs/reproducibility.md`](docs/reproducibility.md) for feature and metric details.

### Review results

Each trained GNN writes JSON metrics. Gather them into a flat CSV:

```bash
python scripts/summarize_results.py experiments --output experiments/summary.csv
```

The helper reads result JSONs recursively and preserves the source path and seed. It does not combine models or seeds statistically; calculate mean and standard deviation only across runs that share the same split, evaluation protocol, and checkpoint-selection rule.

## Reproducibility notes

- Fixed split CSVs are included so the GNN comparisons do not depend on regenerating a randomized split.
- Set seeds explicitly. CUDA, RDKit, PyTorch/PyG versions, and the external pretrained model downloads can still affect exact bitwise repeatability.
- Do not select hyperparameters against the held-out test data. Use validation metrics for selection and evaluate test data once for the final configuration.
- Some comparison code is adapted or wrapped around upstream projects. Check each baseline's folder and the reproduction ledger before interpreting it as a fully independent reproduction.
- Dataset and upstream-model licenses/attribution are described in the relevant folders. Do not redistribute an upstream asset outside its license terms.

## Citation

Please cite the accompanying paper when using this benchmark. The canonical repository is [TheeOliver/B3Pred](https://github.com/TheeOliver/B3Pred).
