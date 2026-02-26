"""
Graph Convolutional Network (GCN) Model

Edge features are transformed into scalar edge weights via a learned MLP,
then used to modulate GCN message passing.

Reference:
    Kipf & Welling "Semi-Supervised Classification with Graph Convolutional Networks" ICLR 2017
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Sequential, Linear, Sigmoid, ReLU
from torch_geometric.nn import GCNConv

from configs.graph_configs import GCNConfig, EDGE_FEATURE_DIM, GRAPH_DESC_DIM
from model.graph_stack import GraphStack


class GCN(GraphStack):
    """
    Graph Convolutional Network with edge weighting.

    Edge features → scalar edge weights (via MLP) → GCN with edge_weight.
    Outputs a pooled embedding; final projection lives in Predictor.
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int = EDGE_FEATURE_DIM,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_graph_attr: bool = True,
            graph_attr_dim: int = GRAPH_DESC_DIM,
            model_name: str = 'GCN',
    ):
        super(GCN, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm,
            use_graph_attr=use_graph_attr,
            graph_attr_dim=graph_attr_dim,
        )

        # Projects edge features to a scalar weight in [0, 1]
        self.edge_mlp = Sequential(
            Linear(edge_dim, edge_dim * 2),
            ReLU(),
            Linear(edge_dim * 2, 1),
            Sigmoid(),
        )

        convs = [
            GCNConv(
                in_channels=node_dim if layer == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
                improved=False,
                add_self_loops=True,
                normalize=True,
            )
            for layer in range(graph_layers)
        ]
        self._convs = ModuleList(convs)

    def _apply_conv(self, layer: int, x, edge_index, edge_attr):
        if edge_attr is None:
            return self._convs[layer](x, edge_index)
        edge_weight = self.edge_mlp(edge_attr).view(-1)
        return self._convs[layer](x, edge_index, edge_weight=edge_weight)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> 'GCN':
        params = {k: config[k] for k in GCNConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', EDGE_FEATURE_DIM)
        return cls(**params)