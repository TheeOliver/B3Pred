"""Graph Attention Network for BBB prediction."""
from typing import Any, Dict
import torch
from torch.nn import ModuleList
from torch_geometric.nn import GATv2Conv
from src.models.base import GraphStack


class GAT(GraphStack):
    """Graph Attention Network (GAT) implementation."""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 3,
        graph_layers: int = 3,
        graph_hidden_channels: int = 64,
        attention_heads: int = 4,
        attention_dropout: float = 0.3,
        graph_dropout: float = 0.3,
        use_graph_norm: bool = True,
        model_name: str = 'GAT'
    ):
        """
        Initialize GAT model.
        
        Args:
            node_dim: Dimension of input node features
            edge_dim: Dimension of edge features
            graph_layers: Number of GAT layers
            graph_hidden_channels: Hidden dimension size
            attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention coefficients
            graph_dropout: Dropout rate after each layer
            use_graph_norm: If True, use GraphNorm; else LayerNorm
            model_name: Name of the model
        """
        super(GAT, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropout=graph_dropout,
            use_graph_norm=use_graph_norm
        )
        
        # Override with GAT convolutions
        convs = [
            GATv2Conv(
                in_channels=graph_hidden_channels if layer != 0 else node_dim,
                out_channels=graph_hidden_channels,
                heads=attention_heads,
                dropout=attention_dropout,
                edge_dim=edge_dim,
                concat=False,  # Average instead of concatenate
            )
            for layer in range(graph_layers)
        ]
        
        self._convs = ModuleList(convs)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], node_dim: int) -> torch.nn.Module:
        """
        Create GAT model from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            node_dim: Dimension of input node features
            
        Returns:
            Initialized GAT model
        """
        return cls(
            node_dim=node_dim,
            graph_layers=config['graph_layers'],
            graph_hidden_channels=config['graph_hidden_channels'],
            attention_heads=config['attention_heads'],
            attention_dropout=config['attention_dropout'],
            graph_dropout=config['graph_dropout'],
            use_graph_norm=config['use_graph_norm'],
            model_name=config['model_name']
        )
