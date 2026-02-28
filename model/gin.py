"""
Graph Isomorphism Network (GIN) Model

GIN is as powerful as the Weisfeiler-Lehman graph isomorphism test.
This variant does NOT use edge features in message passing (see GINE for that).

Reference:
    Xu et al. "How Powerful are Graph Neural Networks?" ICLR 2019
"""

from typing import Any, Dict
import torch
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d, ModuleList
from torch_geometric.nn import GINConv

from configs.graph_configs import GINConfig, GRAPH_DESC_DIM
from model.graph_stack import GraphStack


class GIN(GraphStack):
    """
    Graph Isomorphism Network.

    Does NOT consume edge_attr in message passing.
    graph_attr is still appended after pooling if use_graph_attr=True,
    giving the Predictor access to molecular descriptors.
    """

    def __init__(
            self,
            node_dim: int,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_graph_attr: bool = True,
            graph_attr_dim: int = GRAPH_DESC_DIM,
            model_name: str = 'GIN',
            **kwargs,   # absorb unused keys (e.g. edge_dim) passed from generic config
    ):
        super(GIN, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm,
            use_graph_attr=use_graph_attr,
            graph_attr_dim=graph_attr_dim,
        )

        convs = []
        for layer in range(graph_layers):
            in_channels = node_dim if layer == 0 else graph_hidden_channels
            mlp = Sequential(
                Linear(in_channels, graph_hidden_channels),
                BatchNorm1d(graph_hidden_channels),
                ReLU(),
                Linear(graph_hidden_channels, graph_hidden_channels),
            )
            convs.append(GINConv(mlp, train_eps=True))

        self._convs = ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index)   # edge_attr intentionally ignored
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
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> 'GIN':
        params = {k: config.get(k, v) for k, v in GINConfig.hyperparameters.items()}
        params['node_dim'] = graph_info['node_dim']
        # GIN ignores edge_dim but **kwargs absorbs it safely
        return cls(**params)