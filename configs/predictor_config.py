from typing import Dict, Any
import torch
from configs.graph_configs import (
    GATConfig, GCNConfig, SAGEConfig, GINConfig, GINEConfig,
    GRAPH_DESC_DIM, EDGE_FEATURE_DIM,
)
from model.gat import GAT
from model.gcn import GCN
from model.graphsage import GraphSAGE
from model.gin import GIN
from model.gine import GINE


class PredictorConfig:
    """
    Configuration for the MLP predictor head that sits on top of the graph encoder.

    If use_graph_attr=True (set in the graph config), the pooled graph embedding
    is concatenated with the GRAPH_DESC_DIM-dimensional graph_attr vector before
    being fed into the MLP.  Your predictor MLP in_channels should therefore be:
        graph_hidden_channels + GRAPH_DESC_DIM  (when use_graph_attr=True)
        graph_hidden_channels                   (when use_graph_attr=False)
    """

    hyperparameters: Dict[str, Any] = {
        "pred_layers": 3,
        "pred_hidden_channels": 64,
        "pred_dropouts": 0.3,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf


class GraphConfig():
    models: Dict[str, Any] = {
        "GAT": {
            "model": GAT,
            "config": GATConfig,
        },
        "GCN": {
            "model": GCN,
            "config": GCNConfig,
        },
        "GraphSAGE": {
            "model": GraphSAGE,
            "config": SAGEConfig,
        },
        "GIN": {
            "model": GIN,
            "config": GINConfig,
        },
        "GINE": {
            "model": GINE,
            "config": GINEConfig,
        },
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        return cls.models[config['model_name']]['config'].from_dict(config)


class TrainConfig():

    loss_function: Dict[str, Any] = {
        'crossentropy': torch.nn.CrossEntropyLoss,
    }

    hyperparameters = {
        "subset_size": 0.1,
        "batch_size": 32,
        "epochs": 1,
        "lr": 1e-3,
        "loss": "crossentropy",
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf