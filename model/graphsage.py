"""
GraphSAGE (SAmple and aggreGatE) Model

Learns node embeddings by sampling and aggregating features from local neighbourhoods.
Edge features are projected and added to destination nodes after SAGE aggregation.

Reference:
    Hamilton et al. "Inductive Representation Learning on Large Graphs" NeurIPS 2017
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import SAGEConv

from configs.graph_configs import SAGEConfig, EDGE_FEATURE_DIM, GRAPH_DESC_DIM
from model.graph_stack import GraphStack


class GraphSAGE(GraphStack):
    """
    GraphSAGE with edge feature incorporation.

    Edge features are transformed to graph_hidden_channels and scatter-added
    to destination nodes after SAGE aggregation, enriching node representations
    with local bond information at each layer.
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
            model_name: str = 'GraphSAGE',
    ):
        super(GraphSAGE, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm,
            use_graph_attr=use_graph_attr,
            graph_attr_dim=graph_attr_dim,
        )

        # Projects edge features into node space so they can be added to node embeddings
        self.edge_mlp = Sequential(
            Linear(edge_dim, graph_hidden_channels),
            BatchNorm1d(graph_hidden_channels),
            ReLU(),
            Linear(graph_hidden_channels, graph_hidden_channels),
        )

        convs = [
            SAGEConv(
                in_channels=node_dim if layer == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
                normalize=True,
                root_weight=True,
                project=False,
            )
            for layer in range(graph_layers)
        ]
        self._convs = ModuleList(convs)

    def _apply_conv(self, layer: int, x, edge_index, edge_attr):
        """
        SAGE aggregation followed by edge-feature injection into destination nodes.
        """
        x = self._convs[layer](x, edge_index)

        if edge_attr is not None:
            _, col = edge_index          # col: destination node indices
            edge_emb = self.edge_mlp(edge_attr)
            x.index_add_(0, col, edge_emb)

        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> 'GraphSAGE':
        params = {k: config.get(k, v) for k, v in SAGEConfig.hyperparameters.items()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', EDGE_FEATURE_DIM)
        return cls(**params)