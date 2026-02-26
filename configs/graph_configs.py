from typing import Any, Dict

# New edge feature dimensionality after enrichment (one-hot bond type + flags + one-hot stereo)
# 4 (bond type OH) + 1 (conjugated) + 1 (ring) + 6 (stereo OH) = 12
EDGE_FEATURE_DIM = 12

# Graph-level descriptor dimensionality (MolLogP, TPSA, MolWt, etc.)
GRAPH_DESC_DIM = 14


class GATConfig():
    hyperparameters = {
        'model_name': 'GAT',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'attention_heads': 4,
        'attention_dropouts': 0.3,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        # Set True to concatenate graph_attr to the pooled embedding before the
        # predictor MLP — gives the head direct access to molecular descriptors.
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf


class GCNConfig():
    hyperparameters = {
        "model_name": "GCN",
        "graph_layers": 3,
        "graph_hidden_channels": 256,
        "graph_dropouts": 0.3,
        "graph_norm": True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf


class SAGEConfig():
    hyperparameters = {
        'model_name': 'GraphSAGE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf


class GINConfig():
    hyperparameters = {
        'model_name': 'GIN',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf


class GINEConfig():
    """
    GINE requires explicit edge feature dimensionality because it conditions
    message passing on edge attributes.  Updated to EDGE_FEATURE_DIM=12.
    """
    hyperparameters = {
        'model_name': 'GINE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
        # GINE-specific: must match the enriched edge_attr width
        'edge_dim': EDGE_FEATURE_DIM,
        'use_graph_attr': True,
        'graph_attr_dim': GRAPH_DESC_DIM,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val
        return res_conf