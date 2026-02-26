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

from configs.graph_configs import GATConfig, EDGE_FEATURE_DIM, GRAPH_DESC_DIM
from model.graph_stack import GraphStack


class GAT(GraphStack):
    """
    Graph Attention Network using GATv2Conv.

    Outputs a pooled embedding (+ optional graph_attr concat).
    Final class projection is handled by Predictor.
    """

    def __init__(
            self,
            node_dim: int,
            edge_dim: int = EDGE_FEATURE_DIM,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            attention_heads: int = 4,
            attention_dropouts: float = 0.2,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_graph_attr: bool = True,
            graph_attr_dim: int = GRAPH_DESC_DIM,
            model_name: str = 'GAT',
    ):
        super(GAT, self).__init__(
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
            convs.append(
                GATv2Conv(
                    in_channels=in_channels,
                    out_channels=graph_hidden_channels,
                    heads=attention_heads,
                    dropout=attention_dropouts,
                    edge_dim=edge_dim,
                    concat=False,       # average heads → output is graph_hidden_channels
                    add_self_loops=True,
                    share_weights=False,
                    bias=True,
                )
            )

        self._convs = ModuleList(convs)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch

        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index, edge_attr=edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Pool → [batch_size, graph_hidden_channels]
        x = self.pool(x, batch)

        # Append graph_attr if enabled
        if self.use_graph_attr and hasattr(data, 'graph_attr'):
            graph_attr = data.graph_attr
            if graph_attr.dim() == 3:
                graph_attr = graph_attr.squeeze(1)
            x = torch.cat([x, graph_attr], dim=-1)

        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> 'GAT':
        params = {k: config[k] for k in GATConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        params['edge_dim'] = graph_info.get('edge_dim', EDGE_FEATURE_DIM)
        return cls(**params)