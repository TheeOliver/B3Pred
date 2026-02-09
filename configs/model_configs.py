"""Model configurations for all GNN architectures and training."""
from typing import Any, Dict
import torch


class GATConfig:
    """Graph Attention Network configuration."""
    
    default_params = {
        'model_name': 'GAT',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'attention_heads': 4,
        'attention_dropout': 0.3,
        'graph_dropout': 0.3,
        'use_graph_norm': True,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


class GCNConfig:
    """Graph Convolutional Network configuration."""
    
    default_params = {
        "model_name": "GCN",
        "graph_layers": 3,
        "graph_hidden_channels": 256,
        "graph_dropout": 0.3,
        "use_graph_norm": True,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


class GraphSAGEConfig:
    """GraphSAGE configuration."""
    
    default_params = {
        'model_name': 'GraphSAGE',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropout': 0.3,
        'use_graph_norm': True,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


class GINConfig:
    """Graph Isomorphism Network configuration."""
    
    default_params = {
        'model_name': 'GIN',
        'graph_layers': 3,
        'graph_hidden_channels': 64,
        'graph_dropout': 0.3,
        'use_graph_norm': True,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


class PredictorConfig:
    """Configuration for prediction head."""
    
    default_params = {
        "pred_layers": 3,
        "pred_hidden_channels": 64,
        "pred_dropout": 0.3,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


class TrainingConfig:
    """Training configuration."""
    
    LOSS_FUNCTIONS = {
        'crossentropy': torch.nn.CrossEntropyLoss,
        'bce': torch.nn.BCEWithLogitsLoss,
    }
    
    default_params = {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "loss": "crossentropy",
        "early_stopping_patience": 10,
    }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create config from dictionary, using defaults for missing values."""
        return {key: config.get(key, default_val) 
                for key, default_val in cls.default_params.items()}


# Model registry for easy lookup
MODEL_CONFIGS = {
    "GAT": GATConfig,
    "GCN": GCNConfig,
    "GraphSAGE": GraphSAGEConfig,
    "GIN": GINConfig,
}
