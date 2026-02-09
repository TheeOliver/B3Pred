import json
import argparse
from pathlib import Path
import torch
import numpy as np
from model.gat import GAT
from model.gcn import GCN
from model.gin import GIN
from model.gine import GINE
from model.graphsage import GraphSAGE
from torch_geometric.loader import DataLoader
import pandas as pd
from typing import Dict, Any
from configs.base_config import BaseSettings as settings
from scripts.train import train_model
from scripts.evaluate import test_model
import wandb
from graph.featurizer import MoleculeDataset, compute_feature_stats, normalize_dataset
import itertools

# Available model configurations
MODEL_CONFIGS = {
    'GAT': {
        'model_name': 'GAT',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'attention_heads': 4,
        'attention_dropouts': 0.3,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    },
    'GCN': {
        'model_name': 'GCN',
        'graph_layers': 3,
        'graph_hidden_channels': 256,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    },
    'GraphSAGE': {
        'model_name': 'GraphSAGE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    },
    'GIN': {
        'model_name': 'GIN',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    },
    'GINE': {
        'model_name': 'GINE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    },
}


def get_graph_info(example, hetero=False):
    """Extract graph metadata from a sample"""
    graph_info = {}
    if hetero:
        graph_info['metadata'] = example.metadata()
        graph_info['in_channels'] = example.num_node_features
    else:
        graph_info['node_dim'] = example.x.shape[1]
    return graph_info


def run_training(config: Dict[str, Any], log: bool = False, save: bool = False, test: bool = False):
    """
    Run model training with the given configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary
        log (bool, optional): Enable W&B logging. Defaults to False.
        save (bool, optional): Save model checkpoint. Defaults to False.
        test (bool, optional): Run testing after training. Defaults to False.

    Returns:
        tuple: (results dict, trained model)
    """

    print("\n" + "=" * 70)
    print("TRAINING CONFIGURATION")
    print("=" * 70)
    for key, value in config.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    # Save experiment settings
    if save:
        with open(settings.get_model_folder(config['config_name']) / "config.json", "w") as f:
            json.dump(config, f)

    # Load train and val data
    print("Loading datasets...")
    train = pd.read_csv(settings.TRAIN_DATA)
    val = pd.read_csv(settings.VAL_DATA)

    train_dataset = MoleculeDataset(data=train)
    stats = compute_feature_stats(train_dataset)
    normalize_dataset(train_dataset, stats)

    val_dataset = MoleculeDataset(data=val)
    normalize_dataset(val_dataset, stats)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    # Get graph info
    print("First entry example:")
    print(train_dataset[0])
    graph_info = get_graph_info(train_dataset[0], False)

    # Create model based on config
    print(f"\nBuilding {config['model_name']} model...")
    from configs.predictor_config import GraphConfig
    model_class = GraphConfig.models[config['model_name']]['model']
    pred_model = model_class.from_config(config, graph_info)

    if log:
        wandb.init(project=settings.PROJECT_NAME,
                   config=config, name=config['config_name'])

    # Train model
    print("\nStarting training...")
    res, model = train_model(pred_model,
                             train_loader,
                             val_loader,
                             config['epochs'],
                             [settings.TARGET_LABEL],  # Convert to list
                             loss_type=config['loss'],
                             learning_rate=config['lr'],
                             hetero=False,
                             log=log,
                             save_to=settings.get_model_path(config['config_name']) if save else None)

    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)
    for key, value in res.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    # Test model
    if test:
        res = run_testing(settings.get_model_folder(config['config_name']), log)

    return res, model


def run_testing(model_folder: Path, log: bool = False):
    """
    Run model testing from saved checkpoint.

    Args:
        model_folder (Path): Folder containing model checkpoint
        log (bool, optional): Enable W&B logging. Defaults to False.

    Returns:
        dict: Test results
    """

    # Load config
    with open(model_folder / 'config.json', 'r') as f:
        config = json.load(f)

    # Load test data
    test = pd.read_csv(settings.TEST_DATA)
    test_dataset = MoleculeDataset(data=test)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)

    # Get graph info
    graph_info = get_graph_info(test_dataset[0])

    # Load model
    from configs.predictor_config import GraphConfig
    model_class = GraphConfig.models[config['model_name']]['model']
    model = model_class.from_config(config, graph_info)
    model.load_state_dict(torch.load(model_folder / 'model.pth'))

    # Test model
    res = test_model(test_loader, model, [settings.TEST_LABEL])

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    for key, value in res.items():
        print(f"{key:30s}: {value}")
    print("=" * 70 + "\n")

    # Log results
    if log:
        wandb.log({'evaluation': res})

    return res


def run_grid_search(base_config, search_space, seeds=[0, 1, 2], save_path=None):
    """
    Grid search over hyperparameters with multiple seeds.

    Args:
        base_config (dict): Fixed config values
        search_space (dict): Hyperparameters to search
        seeds (list): Random seeds
        save_path (Path): Optional json save path

    Returns:
        list: All grid search results
    """

    keys, values = zip(*search_space.items())
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

    all_results = []

    print("\n" + "=" * 70)
    print(f"GRID SEARCH: {len(configs)} configurations x {len(seeds)} seeds = {len(configs) * len(seeds)} runs")
    print("=" * 70 + "\n")

    for i, hp_cfg in enumerate(configs):
        print(f"\n===== GRID CONFIG {i + 1}/{len(configs)} =====")
        print(hp_cfg)

        seed_scores = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            config = base_config.copy()
            config.update(hp_cfg)
            config["config_name"] = f"{base_config['model_name']}_gs_{i}_seed{seed}"

            res, _ = run_training(
                config,
                log=False,
                save=False,
                test=False
            )

            seed_scores.append(res["macro_f1"])

        result_entry = {
            **hp_cfg,
            "mean_macro_f1": float(np.mean(seed_scores)),
            "std_macro_f1": float(np.std(seed_scores)),
        }

        all_results.append(result_entry)
        print("RESULT:", result_entry)

    # Sort by mean F1 score
    all_results = sorted(all_results, key=lambda x: x['mean_macro_f1'], reverse=True)

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {save_path}")

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS:")
    print("=" * 70)
    for i, res in enumerate(all_results[:5]):
        print(f"\nRank {i + 1}: F1 = {res['mean_macro_f1']:.4f} ± {res['std_macro_f1']:.4f}")
        print(res)

    return all_results


def create_config_from_args(args):
    """Create configuration dictionary from command line arguments"""

    # Start with base model config
    if args.model in MODEL_CONFIGS:
        config = MODEL_CONFIGS[args.model].copy()
    else:
        raise ValueError(f"Unknown model: {args.model}. Choose from {list(MODEL_CONFIGS.keys())}")

    # Add training parameters
    config.update({
        'config_name': args.name,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'loss': args.loss,
        'pred_layers': args.pred_layers,
        'pred_hidden_channels': args.pred_hidden_channels,
        'pred_dropouts': args.pred_dropouts,
        'subset_size': args.subset_size,
    })

    # Override model-specific parameters if provided
    if args.graph_layers is not None:
        config['graph_layers'] = args.graph_layers
    if args.graph_hidden_channels is not None:
        config['graph_hidden_channels'] = args.graph_hidden_channels
    if args.graph_dropouts is not None:
        config['graph_dropouts'] = args.graph_dropouts
    if args.graph_norm is not None:
        config['graph_norm'] = args.graph_norm

    # GAT-specific parameters
    if args.model == 'GAT':
        if args.attention_heads is not None:
            config['attention_heads'] = args.attention_heads
        if args.attention_dropouts is not None:
            config['attention_dropouts'] = args.attention_dropouts

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Graph Neural Networks for BBB permeability prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train GAT model with default parameters
  python experiments.py --model GAT --name my_gat_experiment --train --save

  # Train GCN with custom parameters
  python experiments.py --model GCN --name custom_gcn --train --epochs 50 --lr 0.0001 --graph_layers 4

  # Test existing model
  python experiments.py --model GAT --name my_gat_experiment --test

  # Run grid search on GINE
  python experiments.py --model GINE --grid_search --name gine_search

  # Train with W&B logging
  python experiments.py --model GAT --name logged_gat --train --save --log
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--train', action='store_true', help='Train the model')
    mode_group.add_argument('--test', action='store_true', help='Test the model')
    mode_group.add_argument('--grid_search', action='store_true', help='Run grid search')

    # Model selection
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--model', type=str, required=True,
                             choices=['GAT', 'GCN', 'GraphSAGE', 'GIN', 'GINE'],
                             help='Graph neural network architecture')
    model_group.add_argument('--name', type=str, required=True,
                             help='Experiment name (used for saving/loading)')

    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    train_group.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    train_group.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    train_group.add_argument('--loss', type=str, default='crossentropy',
                             choices=['crossentropy'], help='Loss function (default: crossentropy)')
    train_group.add_argument('--subset_size', type=float, default=1.0,
                             help='Fraction of data to use (default: 1.0)')

    # Graph model parameters
    graph_group = parser.add_argument_group('Graph Model Parameters')
    graph_group.add_argument('--graph_layers', type=int, help='Number of graph layers')
    graph_group.add_argument('--graph_hidden_channels', type=int, help='Hidden channels in graph layers')
    graph_group.add_argument('--graph_dropouts', type=float, help='Dropout rate in graph layers')
    graph_group.add_argument('--graph_norm', type=lambda x: x.lower() == 'true',
                             help='Use graph normalization (true/false)')

    # GAT-specific parameters
    gat_group = parser.add_argument_group('GAT-Specific Parameters')
    gat_group.add_argument('--attention_heads', type=int, help='Number of attention heads (GAT only)')
    gat_group.add_argument('--attention_dropouts', type=float, help='Attention dropout rate (GAT only)')

    # Predictor parameters
    pred_group = parser.add_argument_group('Predictor Parameters')
    pred_group.add_argument('--pred_layers', type=int, default=2, help='Number of predictor layers (default: 2)')
    pred_group.add_argument('--pred_hidden_channels', type=int, default=64,
                            help='Hidden channels in predictor (default: 64)')
    pred_group.add_argument('--pred_dropouts', type=float, default=0.3,
                            help='Dropout rate in predictor (default: 0.3)')

    # Logging and saving
    misc_group = parser.add_argument_group('Misc Options')
    misc_group.add_argument('--log', action='store_true', help='Enable W&B logging')
    misc_group.add_argument('--save', action='store_true', help='Save model checkpoint')
    misc_group.add_argument('--config_path', type=str, help='Path to custom JSON config file')
    misc_group.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load config from file or create from args
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = create_config_from_args(args)

    # Execute requested mode
    if args.grid_search:
        # Default grid search space
        base_config = config.copy()
        search_space = {
            "graph_layers": [3, 4],
            "graph_hidden_channels": [64, 128],
            "lr": [1e-3, 5e-4, 1e-4],
            "graph_dropouts": [0.0, 0.3, 0.5],
        }

        results = run_grid_search(
            base_config=base_config,
            search_space=search_space,
            seeds=[0, 1, 2],
            save_path=settings.EXPERIMENTS_FOLDER / f"{args.name}_grid_results.json"
        )

    elif args.train:
        res, model = run_training(config, log=args.log, save=args.save, test=args.test)

    elif args.test:
        model_folder = settings.get_model_folder(config['config_name'])
        res = run_testing(model_folder, log=args.log)

    else:
        parser.print_help()
        print("\nError: Please specify at least one mode: --train, --test, or --grid_search")


if __name__ == "__main__":
    main()