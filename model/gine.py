"""
Graph Isomorphism Network with Edge Features (GINE)

GINE extends GIN to incorporate edge features in the message passing.
Useful for molecular graphs where bond information is important.

Reference:
    Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv
from model.graph_stack import GraphStack
from configs.graph_configs import GINEConfig


class GINE(GraphStack):
    """
    Graph Isomorphism Network with Edge Features (GINE).

    Key differences from GIN:
    - Uses edge features in message passing
    - Has separate encoders for nodes and edges
    - Optional ablation studies (can disable node/edge features)
    """

    def __init__(
            self,
            node_dim: int = 9,
            edge_dim: int = 4,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_node_features: bool = True,
            use_edge_features: bool = True,
            model_name: str = "GINE",
    ):
        """
        Initialize GINE model.

        Args:
            node_dim: Size of input node features
            edge_dim: Size of input edge features
            graph_layers: Number of GINE layers
            graph_hidden_channels: Hidden dimension size
            graph_dropouts: Dropout rate
            graph_norm: Use graph normalization
            use_node_features: If False, uses constant node features
            use_edge_features: If False, ignores edge features
            model_name: Model name
        """
        super().__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm,
        )

        self.use_node_features = use_node_features
        self.use_edge_features = use_edge_features

        # Node encoder
        if use_node_features:
            self.node_encoder = Linear(node_dim, graph_hidden_channels)
        else:
            # For ablation: map constant to embedding
            self.node_encoder = Linear(1, graph_hidden_channels)

        # Edge encoder
        if use_edge_features:
            self.edge_encoder = Sequential(
                Linear(edge_dim, graph_hidden_channels),
                BatchNorm1d(graph_hidden_channels),
                ReLU(),
            )
            gine_edge_dim = graph_hidden_channels
        else:
            self.edge_encoder = None
            gine_edge_dim = None

        # Build GINE layers
        convs = []
        for _ in range(graph_layers):
            # MLP for GINE aggregation
            mlp = Sequential(
                Linear(graph_hidden_channels, graph_hidden_channels),
                BatchNorm1d(graph_hidden_channels),
                ReLU(),
                Linear(graph_hidden_channels, graph_hidden_channels),
            )

            # GINE layer with trainable epsilon
            convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=gine_edge_dim,
                    train_eps=True,
                )
            )

        self._convs = ModuleList(convs)

    def forward(self, data):
        """
        Forward pass with edge features.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Graph-level predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode nodes
        if self.use_node_features:
            x = self.node_encoder(x)
        else:
            # Use constant features (for ablation studies)
            ones = torch.ones((x.size(0), 1), device=x.device)
            x = self.node_encoder(ones)

        # Encode edges
        if self.use_edge_features and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)
        else:
            edge_attr = None

        # Pass through GINE layers
        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index, edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Global pooling and prediction
        x = self.pool(x, batch)
        x = self.out_proj(x)
        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]):
        """
        Create GINE model from configuration.

        Args:
            config: Configuration dictionary
            graph_info: Graph metadata (node dimensions, etc.)

        Returns:
            Initialized GINE model
        """
        params = {k: config[k] for k in GINEConfig.hyperparameters.keys()}
        params["node_dim"] = graph_info["node_dim"]
        return cls(**params)