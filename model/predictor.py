"""
Predictor Module

Wraps a GraphStack encoder with an MLP classification head.

The GraphStack outputs a pooled embedding of shape [batch_size, output_dim].
If use_graph_attr=True, output_dim already includes the GRAPH_DESC_DIM
concatenation — no extra wiring needed here.

The MLP head projects: output_dim → pred_hidden_channels (×N layers) → num_classes
"""

import torch
from torch.nn import Dropout, Linear, ModuleList, ReLU
from torch_geometric.nn import LayerNorm
from typing import Any, Dict

from configs.predictor_config import GraphConfig


# BBB is a binary classification task: 0 = non-permeable, 1 = permeable
NUM_CLASSES = 2


class Predictor(torch.nn.Module):
    """
    Graph Prediction Stack: GraphStack encoder + MLP classification head.

    Args:
        graph_model:          A GraphStack subclass instance (GAT, GCN, etc.)
        pred_layers:          Number of MLP layers (including final output layer)
        pred_hidden_channels: Hidden size of intermediate MLP layers
        output_channels:      Number of output classes (default: 2 for BBB)
        pred_dropouts:        Dropout rate in intermediate layers
    """

    def __init__(
            self,
            graph_model: torch.nn.Module,
            pred_layers: int = 3,
            pred_hidden_channels: int = 64,
            output_channels: int = NUM_CLASSES,
            pred_dropouts: float = 0.3,
    ):
        super(Predictor, self).__init__()

        self.graph_stack = graph_model
        self.pred_layers = pred_layers

        # Build MLP layers
        # Layer 0 input: graph_model.output_dim (already includes graph_attr if enabled)
        # Intermediate layers: pred_hidden_channels → pred_hidden_channels
        # Final layer: pred_hidden_channels → output_channels
        self._nns = ModuleList([
            Linear(
                in_features=graph_model.output_dim if i == 0 else pred_hidden_channels,
                out_features=pred_hidden_channels if i != pred_layers - 1 else output_channels,
            )
            for i in range(pred_layers)
        ])

        # Norm + dropout for all layers except the final output layer
        self._norms = ModuleList([
            LayerNorm(pred_hidden_channels)
            for _ in range(pred_layers - 1)
        ])

        self._drops = ModuleList([
            Dropout(pred_dropouts)
            for _ in range(pred_layers - 1)
        ])

        self._act = ReLU()

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        """
        Build a Predictor from a flat config dict + graph_info metadata.

        Args:
            config:     Flat dict with model hyperparams (from graph + predictor configs)
            graph_info: Dict with at least 'node_dim' and 'edge_dim' keys
        """
        graph_model = GraphConfig.models[config['model_name']]['model'].from_config(
            config, graph_info
        )

        return cls(
            graph_model=graph_model,
            pred_layers=config['pred_layers'],
            pred_hidden_channels=config['pred_hidden_channels'],
            output_channels=NUM_CLASSES,
            pred_dropouts=config['pred_dropouts'],
        )

    def forward(self, data):
        """
        Forward pass.

        Args:
            data: PyG Batch / Data object

        Returns:
            Class logits [batch_size, output_channels]
        """
        # Graph encoder → pooled embedding [batch_size, output_dim]
        x = self.graph_stack(data)

        # MLP head
        for i in range(self.pred_layers - 1):
            x = self._nns[i](x)
            x = self._act(x)
            x = self._norms[i](x)
            x = self._drops[i](x)

        # Final linear (no activation — logits for CrossEntropyLoss)
        x = self._nns[-1](x)

        return x