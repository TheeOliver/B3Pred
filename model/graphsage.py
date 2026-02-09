"""
GraphSAGE (SAmple and aggreGatE) Model

GraphSAGE learns node embeddings by sampling and aggregating features from neighbors.
This implementation includes edge feature incorporation.

Reference:
    Hamilton et al. "Inductive Representation Learning on Large Graphs" NeurIPS 2017
"""

from typing import Any, Dict
import torch
from torch.nn import ModuleList, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.nn import SAGEConv
from configs.graph_configs import SAGEConfig
from model.graph_stack import GraphStack


class GraphSAGE(GraphStack):
    """
    GraphSAGE with edge feature incorporation.

    Edge features are processed through an MLP and added to node representations
    after aggregation.
    """

    def __init__(
            self,
            node_dim: int = 9,
            edge_dim: int = 4,
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            model_name: str = 'GraphSAGE'
    ):
        """
        Initialize GraphSAGE model.

        Args:
            node_dim: Size of input node features
            edge_dim: Size of input edge features
            graph_layers: Number of GraphSAGE layers
            graph_hidden_channels: Hidden dimension size
            graph_dropouts: Dropout rate
            graph_norm: Use graph normalization
            model_name: Model name
        """
        super(GraphSAGE, self).__init__(
            model_name=model_name,
            graph_layers=graph_layers,
            graph_hidden_channels=graph_hidden_channels,
            graph_dropouts=graph_dropouts,
            graph_norm=graph_norm
        )

        # Edge feature MLP: transforms edge features to node space
        self.edge_mlp = Sequential(
            Linear(edge_dim, graph_hidden_channels),
            BatchNorm1d(graph_hidden_channels),
            ReLU(),
            Linear(graph_hidden_channels, graph_hidden_channels),
        )

        # Build GraphSAGE layers
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
        Apply GraphSAGE layer with edge feature incorporation.

        The edge features are transformed and added to the destination nodes
        after the SAGE aggregation.

        Args:
            layer: Layer index
            x: Node features
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]

        Returns:
            Updated node features
        """
        # Standard GraphSAGE aggregation
        x = self._convs[layer](x, edge_index)

        # Add edge information if available
        if edge_attr is not None:
            # Get edge endpoints
            row, col = edge_index  # row: source, col: destination

            # Transform edge features
            edge_emb = self.edge_mlp(edge_attr)

            # Add edge embeddings to destination nodes
            # This enriches node representations with edge information
            x.index_add_(0, col, edge_emb)

        return x

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        """
        Create GraphSAGE model from configuration.

        Args:
            config: Configuration dictionary
            graph_info: Graph metadata (node dimensions, etc.)

        Returns:
            Initialized GraphSAGE model
        """
        params = {k: config[k] for k in SAGEConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        return cls(**params)