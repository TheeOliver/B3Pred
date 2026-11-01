# Reproducing the benchmark

This guide describes the empirical work connected to `main.tex`, which is maintained with the manuscript outside this repository. Commands below assume the working directory is the root of this repository.

## Benchmark protocol

- **Primary task:** binary BBB-permeability classification on B3DB.
- **Primary split:** fixed, molecule-similarity/Tanimoto-based train, validation, and test CSVs in `data/`.
- **External evaluation:** MoleculeNet fixed train/validation/test CSVs and LCD/S-data (`data/sdata.csv`, with its model-ready version in `data/prepared/sdata_prepared.csv`). Cross-dataset runs train on B3DB and evaluate the held-out external data unless a baseline section explicitly says otherwise.
- **Columns:** the core B3DB/MoleculeNet files use `SMILES,target`; raw S-data uses `SMILES,BBB` and the prepared copy uses `SMILES,target`.
- **Main metrics:** accuracy, positive-class precision/recall/F1, specificity, ROC-AUC, and Matthews correlation coefficient (MCC). MCC and ROC-AUC should accompany F1 because the external data are imbalanced.
- **GNN feature pipeline:** RDKit molecular graphs with atom/bond features; feature statistics are learned on training data and reused for validation.
- **Selection:** model/HPO selection uses validation data. Test files are for final evaluation only.

The Tanimoto split CSVs are the benchmark inputs. Avoid replacing them with a newly generated random split when comparing with the paper. If rebuilding split files from source datasets, document the source release, preprocessing, similarity threshold, and assignment seed in the experiment record.

## Model inventory and reproducibility level

| Paper model/group | Repository location | What is present | Reproduction status |
| --- | --- | --- | --- |
| GAT, GCN, GraphSAGE, GINE | `model/`, `experiments.py` | In-house implementations, fixed data, HPO scripts, training/evaluation CLI | Trainable locally from source. |
| GIN | `model/gin.py`, `experiments.py` | Additional architecture supported by source and some historical searches | Trainable; not one of the four headline architectures in the paper. |
| LR, RF, XGBoost | `classical_ml.py` | In-house ECFP/Morgan baseline pipeline | Trainable; XGBoost is optional. |
| Atom-Attention-MPNN | `baselines/atom_attention_mpnn/` | Upstream Chemprop-derived source and requirements | Trainable with its own environment and command-line protocol. No upstream training dataset or container image is bundled. |
| Classical ML Descriptor Models (RFC) | `baselines/classical_ml_descriptors/` | Upstream BBB-Models wrapper/source and license | The checked-in paper result used Mordred descriptors and default internal classifiers. Validate dataset/feature setup against the source README before rerunning. |
| RGCN | `baselines/rgcn/` | External RGCN implementation and B3DB evaluation adapter | Source present; requires the upstream protein/drug graph resources and a compatible environment. Large upstream graph/data assets are not bundled. |
| VAE-Embeddings | `baselines/vae_embeddings/` | Evaluation wrapper and VAE/PVAE feature code | The paper evaluation used the documented Morgan fingerprint fallback because a pretrained VAE checkpoint was unavailable. It did not use pretrained VAE embeddings. |
| DeePred-BBB | `baselines/deepred_bbb/` | Published inference script, model checkpoint, and PaDEL descriptor software | Local fixed-checkpoint inference; this is not retraining. Retain PaDEL notices in its folder. The DeePred snapshot itself had no license file. |
| AttentiveFP and GROVER-BBBp | `baselines/bbb_uncertainty_project/` | Relevant source modules and training/evaluation scripts | Paper values came from deterministic inference with fixed pretrained checkpoints. Those checkpoints/results and their acquisition details are not all present here, so exact regeneration requires sourcing the weights described in the upstream project. The local source snapshot had no license file. |
| ChemBERTa-FT and MolCLR | `baselines/chemberta_molclr/` | Experiment scripts | Scripts download/load external pretrained models and require the matching model-specific environment and dataset mapping. Record model revision and tokenizer/model identifiers with each run. |
| LightBBB | — | No local implementation | Paper number was obtained through the public web interface. It cannot be retrained or independently regenerated here. |
| Deep-B3 | — | No local implementation | Paper number was obtained through a public web interface. It cannot be retrained or independently regenerated here. |

The original manuscript has an unfilled supplementary reproduction-details note for the seven published baselines. The copied manuscript is in [`../paper/`](../paper/); several figure assets referenced by its TeX source are missing, as listed in that folder's README. For a new release, record upstream revision, command, configuration, random seed, checkpoint source/hash, failed-SMILES handling, and validation/test protocol in an experiment-specific record before claiming exact reproduction.

## Core GNN workflow

### 1. Environment

Use Python 3.10 or 3.11 for the core pipeline and install the PyTorch/PyG combination suitable for the machine. See [`../requirements.txt`](../requirements.txt) and the official PyG install guide. Install optional HPO dependencies with:

```bash
python -m pip install -r optimizations/requirements_optimization.txt
```

The checked-in dependency files express minimum versions rather than a lockfile. Save `python --version`, `pip freeze`, PyTorch version, PyG version, RDKit version, CUDA version, and GPU model with any new reported run.

### 2. Single training and test run

```bash
python experiments.py --model GINE --name gine_seed42 --train --test --save --seed 42
```

The CLI uses B3DB by default, reads fixed files from `data/`, and writes under `experiments/gine_seed42/`. The result includes the configuration and metrics JSON. For standalone evaluation after training:

```bash
python experiments.py --model GINE --name gine_seed42 --test --results_name b3db_test
python experiments.py --model GINE --name gine_seed42 --test \
  --test_data data/prepared/sdata_prepared.csv \
  --test_target_col target --results_name lcd_test
```

Use distinct `--results_name` values for each test set; otherwise later evaluations overwrite the same JSON path.

To use another output root:

```bash
B3PRED_EXPERIMENTS_ROOT=results/new_study python experiments.py \
  --model GAT --name gat_seed42 --train --test --save --seed 42
```

### 3. Configuration template

`configs/experiment_template.json` lists the fields consumed by the GNN runner. Copy it, update `config_name` and hyperparameters, then pass `--config_path`. Keep `--model` and `--name` consistent with the file's `model_name` and `config_name`. The training interface currently supports the core GNN models; external model scripts have separate options and environments.

### 4. Hyperparameter optimization

All supported search methods are listed by:

```bash
python optimizations/run_all_optimizations.py --help
```

Example one-model, one-method run:

```bash
python optimizations/run_all_optimizations.py \
  --methods cmaes --models GINE --n_trials 20 \
  --subset_size 0.1 --opt_epochs 10 --top_k 3 --full_epochs 100 \
  --results_dir experiments/gine_cmaes
```

The batch runner defaults to 11 methods × 5 model variants, 100 trials, automatic top-k retraining, and test evaluation. Do not launch those defaults as a first run. Choose methods/models and compute budget explicitly. Historical search JSON and checkpoints are under `results/optimization/`.

### 5. Result table

```bash
python scripts/summarize_results.py experiments --output experiments/summary.csv
```

This flattens individual JSON outputs; it does not compute aggregate statistics. For mean ± standard deviation, group only runs that share architecture, dataset split, evaluation set, preprocessing, and checkpoint-selection protocol.

## External baseline runs

External implementations are isolated because they use different dependency stacks, defaults, input conventions, and checkpoint provenance. Keep their outputs under a per-model/per-seed directory and export a common metrics JSON or CSV with model, dataset, seed, metrics, command/config, code revision, and checkpoint hash. The common source CSVs are in `data/`; where the external code expects different columns, make a derived input file in the run's output directory and preserve the original split unchanged.

The baseline folders retain the useful local code and upstream instructions. The original HPC/container scripts from the development workspace contained machine-specific absolute paths and large `.sif` images; those paths/images are deliberately not part of the reusable experiment interface. Use the upstream commands as a starting point, set paths relative to this checkout, and record the environment. Atom-Attention-MPNN and RGCN require GPU-capable environments for paper-scale runs. ChemBERTa/MolCLR require model downloads and network/cache space.

## Result provenance and known differences

Historical artifacts under `results/` are retained from prior runs, not produced by a clean run of this documentation. Some stored optimization paths duplicate earlier artifacts; use the model config and timestamped result file to identify the intended run. The manuscript is authoritative for reported table values and notes the following provenance differences:

- the external descriptor RFC used Mordred 2D descriptors, while `classical_ml.py` uses Morgan fingerprints;
- VAE-Embeddings used Morgan fingerprints rather than a pretrained VAE encoder;
- AttentiveFP and GROVER were fixed-checkpoint inference; LightBBB and Deep-B3 were remote-interface evaluations;
- the externally developed RGCN and Atom-Attention-MPNN were run at their documented/default protocol rather than through the in-house HPO sweep;
- the benchmark reports multiple seeds for several models, but older artifacts in this repository may represent only some seeds/searches.

Therefore, do not treat the existence of an artifact folder as proof that every paper row is independently reproducible from this checkout. Use the table above and archive complete per-run metadata when regenerating values.

## Upstream source revisions

The copied external code was present in the workspace at these upstream revisions. The local Atom-Attention-MPNN, RGCN, BBB uncertainty project, and DeePred-BBB snapshots did not contain a license file; verify permission with their authors or the current upstream repositories before redistributing those folders. Other upstream license/notice files are retained where they were present.

| Component | Upstream | Revision at import |
| --- | --- | --- |
| Atom-Attention-MPNN | [NovaMechanicsOpenSource/Atom-Attention-MPNN](https://github.com/NovaMechanicsOpenSource/Atom-Attention-MPNN) | `eb2e4ad971374a99c67a97e3898d2b405d086ef1` |
| RGCN | [dingyan20/BBB-Penetration-Prediction](https://github.com/dingyan20/BBB-Penetration-Prediction) | `7be28c22548fb0c30bfcb6c452c28da9fde03e7e` |
| BBB-Models | [plissonf/BBB-Models](https://github.com/plissonf/BBB-Models) | `e5c54a706a7b9d769c21f1e50b3ca32ef45d6ec9` |
| VAE-Embeddings | [znavoyan/vae-embeddings](https://github.com/znavoyan/vae-embeddings) | `c8581305248ef9a112a92fb77295151a40785c84` |
| BBB uncertainty project | [tongxiaochu/BBB_uncertainty_project](https://github.com/tongxiaochu/BBB_uncertainty_project) | `5486ab6ab1c14cf5948f2807d3ef7c6529ea60f2` (local checkout had modifications). |
| DeePred-BBB | [12rajnish/DeePred-BBB](https://github.com/12rajnish/DeePred-BBB) | `327e066970f6cb1dd8cf97adb92c16b91675e793` |
| Original B3Pred repository snapshot | [TheeOliver/B3Pred](https://github.com/TheeOliver/B3Pred) | `669bd12b1b519d41a96c55ea1276013bc6085ce9` |

DeePred-BBB and the BBB uncertainty project were local clones without a captured clean revision in this workspace. Their READMEs and license/notices should be checked before modifying or redistributing their assets.
