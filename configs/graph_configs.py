from typing import Any, Dict, cast


class GATConfig():
    hyperparameters = {
        'model_name': 'GAT',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'attention_heads': 4,
        'attention_dropouts': 0.3,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    }

    @classmethod
    def from_dict(cls, config: Dict['str', Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val

        return res_conf

class GCNConfig():
    
    hyperparameters = {
        "model_name":  "GCN",
        "graph_layers": 3,
        "graph_hidden_channels": 256,
        "graph_dropouts": 0.3,
        "graph_norm": True,
    }
    
    @classmethod
    def from_dict(cls, config: Dict['str', Any]):
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
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val

        return res_conf

class GINEConfig():
    hyperparameters = {
        'model_name': 'GINE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropouts': 0.3,
        'graph_norm': True,
    }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]):
        res_conf = {}
        for key, val in cls.hyperparameters.items():
            res_conf[key] = type(val)(config[key]) if key in config else val

        return res_conf