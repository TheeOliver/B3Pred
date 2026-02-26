"""
Base Graph Stack Module

Provides a common base class for all GNN architectures.
Handles normalization, dropout, pooling, and (optionally) graph_attr concatenation.

NOTE: Output projection (graph embedding → class logits) lives in Predictor, not here.
GraphStack outputs a pooled embedding of shape [batch_size, output_dim].
"""

from typing import Any, Dict
import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm, global_mean_pool

from configs.graph_configs import GRAPH_DESC_DIM


class GraphStack(torch.nn.Module):
    """
    Base class for Graph Neural Network stacks.

    Outputs a pooled graph-level embedding.  If use_graph_attr=True, the
    GRAPH_DESC_DIM-dimensional graph_attr tensor stored on each Data object is
    concatenated to the pooled embedding, giving the downstream Predictor direct
    access to physicochemical molecular descriptors without them getting averaged
    away during pooling.

    output_dim reflects the actual embedding size:
        graph_hidden_channels                  (use_graph_attr=False)
        graph_hidden_channels + GRAPH_DESC_DIM (use_graph_attr=True)
    """

    def __init__(
            self,
            model_name: str = 'base',
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True,
            use_graph_attr: bool = True,
            graph_attr_dim: int = GRAPH_DESC_DIM,
    ):
        super(GraphStack, self).__init__()

        self.model_name = model_name
        self.graph_layers = graph_layers
        self.graph_hidden_channels = graph_hidden_channels
        self.use_graph_attr = use_graph_attr
        self.graph_attr_dim = graph_attr_dim if use_graph_attr else 0

        # output_dim is what Predictor reads to size its first linear layer
        self.output_dim = graph_hidden_channels + self.graph_attr_dim

        # Placeholder convs (overridden by child classes)
        convs = [
            GCNConv(
                in_channels=-1 if i == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
            )
            for i in range(graph_layers)
        ]

        # Normalization layers
        norms = [
            GraphNorm(in_channels=graph_hidden_channels) if graph_norm
            else LayerNorm(in_channels=graph_hidden_channels, mode="node")
            for _ in range(graph_layers)
        ]

        # Dropout layers
        drops = [Dropout(p=graph_dropouts) for _ in range(graph_layers)]

        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)

        self.pool = global_mean_pool

    def forward(self, data):
        """
        Forward pass through the graph stack.

        Returns:
            Pooled embedding [batch_size, output_dim]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch

        for i in range(self.graph_layers):
            x = self._apply_conv(i, x, edge_index, edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Global pooling → [batch_size, graph_hidden_channels]
        x = self.pool(x, batch)

        # Concatenate graph-level descriptors if requested
        if self.use_graph_attr and hasattr(data, 'graph_attr'):
            # graph_attr shape per graph: (1, GRAPH_DESC_DIM) → squeeze to (GRAPH_DESC_DIM,)
            # After batching with DataLoader it becomes (batch_size, GRAPH_DESC_DIM)
            graph_attr = data.graph_attr
            if graph_attr.dim() == 3:
                # shape: (batch_size, 1, GRAPH_DESC_DIM) — squeeze middle dim
                graph_attr = graph_attr.squeeze(1)
            x = torch.cat([x, graph_attr], dim=-1)

        return x

    def _apply_conv(self, layer: int, x, edge_index, edge_attr):
        """
        Apply convolution layer.  Child classes override for custom edge handling.
        """
        return self._convs[layer](x, edge_index)

    def reset_parameters(self):
        for conv in self._convs:
            conv.reset_parameters()
        for norm in self._norms:
            norm.reset_parameters()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_name={self.model_name},\n"
            f"  layers={self.graph_layers},\n"
            f"  hidden_channels={self.graph_hidden_channels},\n"
            f"  output_dim={self.output_dim},\n"
            f"  use_graph_attr={self.use_graph_attr}\n"
            f")"
        )