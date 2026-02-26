import json
import argparse
from pathlib import Path
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import pandas as pd
from typing import Dict, Any
import itertools

from configs.base_config import BaseSettings as settings
from configs.graph_configs import EDGE_FEATURE_DIM, GRAPH_DESC_DIM
from model.predictor import Predictor
from scripts.train import train_model
from scripts.evaluate import test_model
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Canonical model configs
#  use_graph_attr / graph_attr_dim / edge_dim are now explicit so they survive
#  round-trips through JSON (grid search saves + reloads configs).
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    'GAT': {
        'model_name': 'GAT',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'attention_heads': 4,
        'attention_dropouts': 0.3,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    },
    'GCN': {
        'model_name': 'GCN',
        'graph_layers': 3,
        'graph_hidden_channels': 256,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    },
    'GraphSAGE': {
        'model_name': 'GraphSAGE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    },
    'GIN': {
        'model_name': 'GIN',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    },
    'GINE': {
        'model_name': 'GINE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'edge_dim': EDGE_FEATURE_DIM,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    },
}


def get_graph_info(example) -> Dict[str, Any]:
    """
    Extract graph metadata from a single Data object.

    Returns:
        Dict with 'node_dim' and 'edge_dim' keys, used by all from_config methods.
    """
    graph_info = {
        'node_dim': example.x.shape[1],
        'edge_dim': example.edge_attr.shape[1] if example.edge_attr is not None else EDGE_FEATURE_DIM,
    }
    return graph_info


def _build_datasets(config: Dict[str, Any]):
    """
    Load and normalize train + val datasets.  Returns datasets, loaders, stats.
    """
    train_df = pd.read_csv(settings.TRAIN_DATA)
    val_df   = pd.read_csv(settings.VAL_DATA)

    train_dataset = MoleculeDataset(data=train_df)
    stats = compute_feature_stats(train_dataset)
    normalize_dataset(train_dataset, stats)

    val_dataset = MoleculeDataset(data=val_df)
    normalize_dataset(val_dataset, stats)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'], shuffle=False)

    return train_dataset, val_dataset, train_loader, val_loader, stats


def run_training(config: Dict[str, Any], log: bool = False, save: bool = False, test: bool = False):
    """
    Run model training with the given configuration.

    Args:
        config: Flat config dict (graph + predictor + training hyperparams)
        log:    Enable W&B logging
        save:   Save model checkpoint + stats to experiment folder
        test:   Run test set evaluation after training

    Returns:
        (results dict, trained Predictor model)
    """
    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    model_folder = settings.get_model_folder(config['config_name'])

    if save:
        with open(model_folder / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("Loading datasets...")
    train_dataset, val_dataset, train_loader, val_loader, stats = _build_datasets(config)

    # Persist normalisation stats alongside the model so run_testing can reuse them
    if save:
        torch.save(stats, model_folder / "feature_stats.pt")

    # ── Graph metadata ─────────────────────────────────────────────────────────
    example = train_dataset[0]
    print(f"Example graph: {example}")
    graph_info = get_graph_info(example)
    print(f"Graph info: {graph_info}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {config['model_name']} model...")
    pred_model = Predictor.from_config(config, graph_info)
    print(pred_model)

    if log and WANDB_AVAILABLE:
        wandb.init(project=settings.PROJECT_NAME, config=config, name=config['config_name'])
    elif log:
        print("Warning: wandb not installed — logging disabled.")

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    res, model = train_model(
        pred_model,
        train_loader,
        val_loader,
        config['epochs'],
        [settings.TARGET_LABEL],
        loss_type=config['loss'],
        learning_rate=config['lr'],
        hetero=False,
        log=log,
        save_to=settings.get_model_path(config['config_name']) if save else None,
    )

    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    for key, value in res.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    if test:
        res = run_testing(model_folder, log=log)

    return res, model


def run_testing(model_folder: Path, log: bool = False):
    """
    Run test-set evaluation from a saved checkpoint.

    Reuses the feature normalisation stats saved during training so the test
    set is processed identically to the training set.

    Args:
        model_folder: Folder containing config.json, model.pth, feature_stats.pt
        log:          Enable W&B logging

    Returns:
        dict of test metrics
    """
    model_folder = Path(model_folder)

    with open(model_folder / 'config.json', 'r') as f:
        config = json.load(f)

    # ── Test dataset ──────────────────────────────────────────────────────────
    test_df = pd.read_csv(settings.TEST_DATA)
    test_dataset = MoleculeDataset(data=test_df)

    # Load and apply the same stats used during training
    stats_path = model_folder / "feature_stats.pt"
    if stats_path.exists():
        stats = torch.load(stats_path)
        normalize_dataset(test_dataset, stats)
    else:
        print("Warning: feature_stats.pt not found — test set will NOT be normalised.")

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    graph_info = get_graph_info(test_dataset[0])
    model = Predictor.from_config(config, graph_info)
    model.load_state_dict(torch.load(model_folder / 'model.pth', map_location='cpu'))

    # ── Evaluate ──────────────────────────────────────────────────────────────
    res = test_model(test_loader, model, [settings.TEST_LABEL])

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    for key, value in res.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    if log and WANDB_AVAILABLE:
        wandb.log({'evaluation': res})

    return res


def run_grid_search(base_config, search_space, seeds=(0, 1, 2), save_path=None):
    """
    Grid search over hyperparameters with multiple random seeds.

    Args:
        base_config:  Fixed config values (model_name, pred_*, batch_size, etc.)
        search_space: Dict mapping param name → list of values to try
        seeds:        Random seeds for reproducibility
        save_path:    Optional Path to write JSON results

    Returns:
        Sorted list of result dicts (best first by mean macro F1)
    """
    keys, values = zip(*search_space.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"\nGRID SEARCH: {len(configs)} configs × {len(seeds)} seeds = {len(configs) * len(seeds)} runs\n")

    all_results = []

    for i, hp_cfg in enumerate(configs):
        print(f"===== GRID CONFIG {i + 1}/{len(configs)} =====")
        print(hp_cfg)

        seed_scores = []
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = base_config.copy()
            config.update(hp_cfg)
            config["config_name"] = f"{base_config['model_name']}_gs_{i}_seed{seed}"

            res, _ = run_training(config, log=False, save=False, test=False)
            seed_scores.append(res["macro_f1"])

        result_entry = {
            **hp_cfg,
            "mean_macro_f1": float(np.mean(seed_scores)),
            "std_macro_f1":  float(np.std(seed_scores)),
        }
        all_results.append(result_entry)
        print("RESULT:", result_entry)

    all_results = sorted(all_results, key=lambda x: x['mean_macro_f1'], reverse=True)

    if save_path is not None:
        save_path = Path(save_path)
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    print("\nTOP 5 CONFIGURATIONS:")
    for i, res in enumerate(all_results[:5]):
        print(f"\nRank {i + 1}: F1 = {res['mean_macro_f1']:.4f} ± {res['std_macro_f1']:.4f}")
        print(res)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def create_config_from_args(args) -> Dict[str, Any]:
    """Build a flat config dict from parsed CLI arguments."""
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[args.model].copy()
    config.update({
        'config_name':          args.name,
        'batch_size':           args.batch_size,
        'epochs':               args.epochs,
        'lr':                   args.lr,
        'loss':                 args.loss,
        'pred_layers':          args.pred_layers,
        'pred_hidden_channels': args.pred_hidden_channels,
        'pred_dropouts':        args.pred_dropouts,
        'subset_size':          args.subset_size,
    })

    # Optional overrides
    if args.graph_layers          is not None: config['graph_layers']          = args.graph_layers
    if args.graph_hidden_channels is not None: config['graph_hidden_channels'] = args.graph_hidden_channels
    if args.graph_dropouts        is not None: config['graph_dropouts']        = args.graph_dropouts
    if args.graph_norm            is not None: config['graph_norm']            = args.graph_norm

    if args.model == 'GAT':
        if args.attention_heads    is not None: config['attention_heads']    = args.attention_heads
        if args.attention_dropouts is not None: config['attention_dropouts'] = args.attention_dropouts

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate GNNs for BBB permeability prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments.py --model GAT --name my_gat --train --save
  python experiments.py --model GCN --name custom_gcn --train --epochs 50 --lr 0.0001
  python experiments.py --model GAT --name my_gat --test
  python experiments.py --model GINE --name gine_search --grid_search
  python experiments.py --model GAT --name logged_gat --train --save --log
        """
    )

    mode = parser.add_argument_group('Mode')
    mode.add_argument('--train',       action='store_true')
    mode.add_argument('--test',        action='store_true')
    mode.add_argument('--grid_search', action='store_true')

    mdl = parser.add_argument_group('Model')
    mdl.add_argument('--model', type=str, required=True,
                     choices=list(MODEL_CONFIGS.keys()))
    mdl.add_argument('--name', type=str, required=True)

    trn = parser.add_argument_group('Training')
    trn.add_argument('--batch_size',   type=int,   default=64)
    trn.add_argument('--epochs',       type=int,   default=20)
    trn.add_argument('--lr',           type=float, default=1e-3)
    trn.add_argument('--loss',         type=str,   default='crossentropy', choices=['crossentropy'])
    trn.add_argument('--subset_size',  type=float, default=1.0)

    grph = parser.add_argument_group('Graph model')
    grph.add_argument('--graph_layers',          type=int)
    grph.add_argument('--graph_hidden_channels', type=int)
    grph.add_argument('--graph_dropouts',        type=float)
    grph.add_argument('--graph_norm',            type=lambda x: x.lower() == 'true')

    gat = parser.add_argument_group('GAT-specific')
    gat.add_argument('--attention_heads',    type=int)
    gat.add_argument('--attention_dropouts', type=float)

    pred = parser.add_argument_group('Predictor')
    pred.add_argument('--pred_layers',          type=int,   default=2)
    pred.add_argument('--pred_hidden_channels', type=int,   default=64)
    pred.add_argument('--pred_dropouts',        type=float, default=0.3)

    misc = parser.add_argument_group('Misc')
    misc.add_argument('--log',         action='store_true')
    misc.add_argument('--save',        action='store_true')
    misc.add_argument('--config_path', type=str)
    misc.add_argument('--seed',        type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = create_config_from_args(args)

    if args.grid_search:
        base_config = config.copy()
        search_space = {
            "graph_layers":          [3, 4],
            "graph_hidden_channels": [64, 128],
            "lr":                    [1e-3, 5e-4, 1e-4],
            "graph_dropouts":        [0.0, 0.3, 0.5],
        }
        run_grid_search(
            base_config=base_config,
            search_space=search_space,
            seeds=[0, 1, 2],
            save_path=settings.EXPERIMENTS_FOLDER / f"{args.name}_grid_results.json",
        )

    elif args.train:
        run_training(config, log=args.log, save=args.save, test=args.test)

    elif args.test:
        model_folder = settings.get_model_folder(config['config_name'])
        run_testing(model_folder, log=args.log)

    else:
        parser.print_help()
        print("\nError: specify --train, --test, or --grid_search")


if __name__ == "__main__":
    main()