"""
Graph Isomorphism Network with Edge Features (GINE)

GINE extends GIN to incorporate edge features in message passing.

Reference:
    Hu et al. "Strategies for Pre-training Graph Neural Networks" ICLR 2020
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Sequential, Linear, ReLU, BatchNorm1d
from torch_geometric.nn import GINEConv

from configs.graph_configs import GINEConfig, EDGE_FEATURE_DIM, GRAPH_DESC_DIM
from model.graph_stack import GraphStack


class GINE(GraphStack):
    """
    Graph Isomorphism Network with Edge Features.

    Node encoder projects raw node features to graph_hidden_channels before
    message passing so all GINE layers operate in a fixed-width space.

    Edge encoder projects the 12-dim one-hot bond features to graph_hidden_channels
    so edge and node tensors are compatible inside GINEConv.
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int = EDGE_FEATURE_DIM,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_node_features: bool = True,
            use_edge_features: bool = True,
            use_graph_attr: bool = True,
            graph_attr_dim: int = GRAPH_DESC_DIM,
            model_name: str = 'GINE',
    ):
        super().__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm,
            use_graph_attr=use_graph_attr,
            graph_attr_dim=graph_attr_dim,
        )

        self.use_node_features = use_node_features
        self.use_edge_features = use_edge_features

        # Node encoder: raw node_dim → graph_hidden_channels
        self.node_encoder = Linear(
            node_dim if use_node_features else 1,
            graph_hidden_channels,
        )

        # Edge encoder: edge_dim → graph_hidden_channels (must match node width for GINEConv)
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

        # GINE layers (all operate in graph_hidden_channels space)
        convs = []
        for _ in range(graph_layers):
            mlp = Sequential(
                Linear(graph_hidden_channels, graph_hidden_channels),
                BatchNorm1d(graph_hidden_channels),
                ReLU(),
                Linear(graph_hidden_channels, graph_hidden_channels),
            )
            convs.append(GINEConv(nn=mlp, edge_dim=gine_edge_dim, train_eps=True))

        self._convs = ModuleList(convs)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Encode nodes
        if self.use_node_features:
            x = self.node_encoder(x)
        else:
            ones = torch.ones((x.size(0), 1), device=x.device)
            x = self.node_encoder(ones)

        # Encode edges
        edge_attr = None
        if self.use_edge_features and data.edge_attr is not None:
            edge_attr = self.edge_encoder(data.edge_attr)

        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index, edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        x = self.pool(x, batch)

        if self.use_graph_attr and hasattr(data, 'graph_attr'):
            graph_attr = data.graph_attr
            if graph_attr.dim() == 3:
                graph_attr = graph_attr.squeeze(1)
            x = torch.cat([x, graph_attr], dim=-1)

        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> 'GINE':
        params = {k: config[k] for k in GINEConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', EDGE_FEATURE_DIM)
        return cls(**params)