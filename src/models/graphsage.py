"""GraphSAGE for BBB prediction."""
from typing import Any, Dict
import torch
from torch.nn import ModuleList
from torch_geometric.nn import SAGEConv
from src.models.base import GraphStack


class GraphSAGE(GraphStack):
    """GraphSAGE implementation."""
    
    def __init__(
        self,
        node_dim: int,
        graph_layers: int = 3,
        graph_hidden_channels: int = 64,
        graph_dropout: float = 0.3,
        use_graph_norm: bool = True,
        model_name: str = 'GraphSAGE'
    ):
        """
        Initialize GraphSAGE model.
        
        Args:
            node_dim: Dimension of input node features
            graph_layers: Number of SAGE layers
            graph_hidden_channels: Hidden dimension size
            graph_dropout: Dropout rate after each layer
            use_graph_norm: If True, use GraphNorm; else LayerNorm
            model_name: Name of the model
        """
        super(GraphSAGE, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropout=graph_dropout,
            use_graph_norm=use_graph_norm
        )
        
        # Override with SAGE convolutions
        convs = [
            SAGEConv(
                in_channels=graph_hidden_channels if layer != 0 else node_dim,
                out_channels=graph_hidden_channels,
            )
            for layer in range(graph_layers)
        ]
        
        self._convs = ModuleList(convs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], node_dim: int) -> torch.nn.Module:
        """
        Create GraphSAGE model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            node_dim: Dimension of input node features
            
        Returns:
            Initialized GraphSAGE model
        """
        return cls(
            node_dim=node_dim,
            graph_layers=config['graph_layers'],
            graph_hidden_channels=config['graph_hidden_channels'],
            graph_dropout=config['graph_dropout'],
            use_graph_norm=config['use_graph_norm'],
            model_name=config['model_name']
        )
