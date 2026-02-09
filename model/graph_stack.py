"""
Base Graph Stack Module

Provides a common base class for all GNN architectures.
Handles normalization, dropout, pooling, and output projection.
"""

from typing import Any, Dict
import torch
from torch.nn import Dropout, ModuleList, Linear
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm, global_mean_pool


class GraphStack(torch.nn.Module):
    """
    Base class for Graph Neural Network stacks.

    Provides common functionality:
    - Normalization layers (GraphNorm or LayerNorm)
    - Dropout layers
    - Global pooling
    - Output projection

    Child classes should override:
    - __init__: to define their specific convolution layers
    - forward: if they need custom forward logic
    """

    def __init__(
            self,
            model_name: str = 'base',
            graph_layers: int = 3,
            graph_hidden_channels: int = 64,
            graph_dropouts: float = 0.5,
            graph_norm: bool = True
    ):
        """
        Initialize base graph stack.

        Args:
            model_name: Name of the model
            graph_layers: Number of graph convolution layers
            graph_hidden_channels: Hidden dimension size
            graph_dropouts: Dropout rate (0.0 to 1.0)
            graph_norm: If True, use GraphNorm; else use LayerNorm
        """
        super(GraphStack, self).__init__()

        self.model_name = model_name
        self.graph_layers = graph_layers
        self.graph_hidden_channels = graph_hidden_channels
        self.output_dim = graph_hidden_channels

        # Initialize convolution layers (child classes override this)
        convs = []
        for i in range(graph_layers):
            convs.append(
                GCNConv(
                    in_channels=-1 if i == 0 else graph_hidden_channels,
                    out_channels=graph_hidden_channels,
                )
            )

        # Normalization layers
        norms = []
        for _ in range(graph_layers):
            if graph_norm:
                norms.append(GraphNorm(in_channels=graph_hidden_channels))
            else:
                norms.append(LayerNorm(in_channels=graph_hidden_channels, mode="node"))

        # Dropout layers
        drops = [Dropout(p=graph_dropouts) for _ in range(graph_layers)]

        # Store as ModuleLists
        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)

        # Pooling and output
        self.pool = global_mean_pool
        self.out_proj = Linear(graph_hidden_channels, 2)  # Binary classification

    def forward(self, data):
        """
        Forward pass through the graph stack.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, node_dim]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim] (optional)
                - batch: Batch assignment [num_nodes]

        Returns:
            Graph-level predictions [batch_size, num_classes]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = data.batch

        # Pass through graph layers
        for i in range(self.graph_layers):
            x = self._apply_conv(i, x, edge_index, edge_attr)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Global pooling: [num_nodes, hidden_dim] -> [batch_size, hidden_dim]
        x = self.pool(x, batch)

        # Output projection: [batch_size, hidden_dim] -> [batch_size, num_classes]
        x = self.out_proj(x)

        return x

    def _apply_conv(self, layer: int, x, edge_index, edge_attr):
        """
        Apply convolution layer.

        Child classes can override this to handle edge attributes differently.

        Args:
            layer: Layer index
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes (may be None)

        Returns:
            Updated node features
        """
        return self._convs[layer](x, edge_index)

    def reset_parameters(self):
        """Reset all learnable parameters."""
        for conv in self._convs:
            conv.reset_parameters()
        for norm in self._norms:
            norm.reset_parameters()
        self.out_proj.reset_parameters()

    def __repr__(self):
        """String representation of the model."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  model_name={self.model_name},\n"
            f"  layers={self.graph_layers},\n"
            f"  hidden_channels={self.graph_hidden_channels},\n"
            f"  output_dim={self.output_dim}\n"
            f")"
        )