"""
Graph Convolutional Network (GCN) Model

GCN uses spectral graph convolutions based on Chebyshev polynomials.
This implementation includes edge weighting via a learned MLP.

Reference:
    Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" ICLR 2017
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Sequential, Linear, Sigmoid
from torch_geometric.nn import GCNConv
from configs.graph_configs import GCNConfig
from model.graph_stack import GraphStack


class GCN(GraphStack):
    """
    Graph Convolutional Network with edge weighting.

    Edge features are transformed into edge weights via an MLP,
    then used in the GCN message passing.
    """

    def __init__(
            self,
            node_dim: int = 9,
            edge_dim: int = 4,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            model_name: str = 'GCN'
    ):
        """
        Initialize GCN model.

        Args:
            node_dim: Size of input node features
            edge_dim: Size of input edge features
            graph_layers: Number of GCN layers
            graph_hidden_channels: Hidden dimension size
            graph_dropouts: Dropout rate
            graph_norm: Use graph normalization
            model_name: Model name
        """
        super(GCN, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm
        )

        # Edge feature MLP: transforms edge features to scalar weights
        self.edge_mlp = Sequential(
            Linear(edge_dim, edge_dim * 2),
            torch.nn.ReLU(),
            Linear(edge_dim * 2, 1),
            Sigmoid()  # Weights in [0, 1]
        )

        # Build GCN layers
        convs = [
            GCNConv(
                in_channels=node_dim if layer == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
                improved=False,  # Use standard GCN
                add_self_loops=True,
                normalize=True,
            )
            for layer in range(graph_layers)
        ]

        self._convs = ModuleList(convs)

    def _apply_conv(self, layer: int, x, edge_index, edge_attr):
        """
        Apply GCN layer with edge weighting.

        Args:
            layer: Layer index
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features (may be None)

        Returns:
            Updated node features
        """
        if edge_attr is None:
            # No edge features: standard GCN
            return self._convs[layer](x, edge_index)

        # Transform edge features to weights
        edge_weight = self.edge_mlp(edge_attr).view(-1)
        return self._convs[layer](x, edge_index, edge_weight=edge_weight)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        """
        Create GCN model from configuration.

        Args:
            config: Configuration dictionary
            graph_info: Graph metadata (node dimensions, etc.)

        Returns:
            Initialized GCN model
        """
        params = {k: config[k] for k in GCNConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', 4)  # Default to 4
        return cls(**params)
