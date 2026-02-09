"""
Graph Isomorphism Network (GIN) Model

GIN is designed to be as powerful as the Weisfeiler-Lehman graph isomorphism test.
This implementation does NOT use edge features (use GINE for edge features).

Reference:
    Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019
"""

from typing import Any, Dict
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, ModuleList
from torch_geometric.nn import GINConv
from configs.graph_configs import GINConfig
from model.graph_stack import GraphStack


class GIN(GraphStack):
    """
    Graph Isomorphism Network (GIN) implementation.

    GIN does NOT use edge features - it only considers graph structure and node features.
    For edge-informed GIN, use GINE instead.
    """

    def __init__(
            self,
            node_dim: int = 9,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            model_name: str = 'GIN'
    ):
        """
        Initialize GIN model.

        Args:
            node_dim: Size of input node features
            graph_layers: Number of GIN layers
            graph_hidden_channels: Hidden dimension size
            graph_dropouts: Dropout rate
            graph_norm: Use graph normalization
            model_name: Model name
        """
        super(GIN, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm
        )

        # Build GIN layers
        convs = []
        for layer in range(graph_layers):
            in_channels = node_dim if layer == 0 else graph_hidden_channels

            # MLP for GIN aggregation
            mlp = Sequential(
                Linear(in_channels, graph_hidden_channels),
                BatchNorm1d(graph_hidden_channels),
                ReLU(),
                Linear(graph_hidden_channels, graph_hidden_channels),
            )

            # GIN layer with trainable epsilon
            convs.append(GINConv(mlp, train_eps=True))

        self._convs = ModuleList(convs)

    def forward(self, data):
        """
        Forward pass - GIN ignores edge attributes.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Graph-level predictions
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # Pass through GIN layers (edge_attr is ignored)
        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Global pooling and prediction
        x = self.pool(x, batch)
        x = self.out_proj(x)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        """
        Create GIN model from configuration.

        Args:
            config: Configuration dictionary
            graph_info: Graph metadata (node dimensions, etc.)

        Returns:
            Initialized GIN model
        """
        params = {k: config[k] for k in GINConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        return cls(**params)