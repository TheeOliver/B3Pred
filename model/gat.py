"""
Graph Attention Network (GAT) Model

GAT uses attention mechanisms to learn importance weights for neighbors.
This implementation uses GATv2Conv for improved expressiveness.

Reference:
    Veličković et al. "Graph Attention Networks" ICLR 2018
    Brody et al. "How Attentive are Graph Attention Networks?" ICLR 2022 (GATv2)
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList
from torch_geometric.nn import GATv2Conv
from configs.graph_configs import GATConfig
from model.graph_stack import GraphStack


class GAT(GraphStack):
    """
    Graph Attention Network using GATv2Conv.

    Features:
    - Multi-head attention
    - Edge features incorporated in attention
    - Attention dropout for regularization
    """

    def __init__(
            self,
            node_dim: int = 9,
            edge_dim: int = 4,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            attention_heads: int = 4,
            attention_dropouts: float = 0.2,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            model_name: str = 'GAT'
    ):
        """
        Initialize GAT model.

        Args:
            node_dim: Size of input node features
            edge_dim: Size of input edge features
            graph_layers: Number of GAT layers
            graph_hidden_channels: Hidden dimension size (per head)
            attention_heads: Number of attention heads
            attention_dropouts: Dropout rate for attention coefficients
            graph_dropouts: Dropout rate after normalization
            graph_norm: Use graph normalization
            model_name: Model name
        """
        super(GAT, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm
        )

        # Build GAT layers
        convs = []
        for layer in range(graph_layers):
            in_channels = node_dim if layer == 0 else graph_hidden_channels

            convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=graph_hidden_channels,
                    heads=attention_heads,
                    dropout=attention_dropouts,
                    edge_dim=edge_dim,
                    concat=False,  # Average attention heads instead of concatenating
                    add_self_loops=True,
                    share_weights=False,
                    bias=True,
                )
            )

        self._convs = ModuleList(convs)

    def forward(self, data):
        """
        Forward pass with attention.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Graph-level predictions
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch

        # Pass through GAT layers
        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index, edge_attr=edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Global pooling and prediction
        x = self.pool(x, batch)
        x = self.out_proj(x)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        """
        Create GAT model from configuration.

        Args:
            config: Configuration dictionary
            graph_info: Graph metadata (node dimensions, etc.)

        Returns:
            Initialized GAT model
        """
        params = {k: config[k] for k in GATConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', 4)  # Default to 4
        return cls(**params)
