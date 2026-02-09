"""Base graph neural network stack for BBB prediction."""
import torch
from torch.nn import Dropout, ModuleList
from torch_geometric.nn import GCNConv, LayerNorm, GraphNorm
from torch_geometric.nn import global_mean_pool


class GraphStack(torch.nn.Module):
    """
    Base class for graph neural network stacks.
    
    Provides common functionality for graph convolution layers,
    normalization, and dropout. Child classes should override
    the _convs ModuleList with their specific convolution type.
    """
    
    def __init__(
        self,
        model_name: str = 'GraphStack',
        graph_layers: int = 3,
        graph_hidden_channels: int = 64,
        graph_dropout: float = 0.5,
        use_graph_norm: bool = True
    ):
        """
        Initialize the graph stack.
        
        Args:
            model_name: Name of the model
            graph_layers: Number of graph convolution layers
            graph_hidden_channels: Hidden dimension size
            graph_dropout: Dropout rate after each layer
            use_graph_norm: If True, use GraphNorm; else LayerNorm
        """
        super(GraphStack, self).__init__()
        
        self.model_name = model_name
        self.graph_layers = graph_layers
        self.output_dim = graph_hidden_channels
        
        # Initialize normalization and dropout layers
        norms = []
        for _ in range(graph_layers):
            if use_graph_norm:
                norms.append(GraphNorm(in_channels=graph_hidden_channels))
            else:
                norms.append(LayerNorm(in_channels=graph_hidden_channels, mode="node"))
        
        drops = [Dropout(p=graph_dropout) for _ in range(graph_layers)]
        
        # Default convolutions (will be overridden by child classes)
        convs = [
            GCNConv(
                in_channels=-1 if i == 0 else graph_hidden_channels,
                out_channels=graph_hidden_channels,
            )
            for i in range(graph_layers)
        ]
        
        self._convs = ModuleList(convs)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)
    
    def forward(self, data):
        """
        Forward pass through graph layers.
        
        Args:
            data: PyTorch Geometric data object with x, edge_index, batch
            
        Returns:
            Graph-level embeddings of shape [num_graphs, output_dim]
        """
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # Apply graph convolution layers
        for i in range(self.graph_layers):
            x = self._convs[i](x, edge_index)
            x = self._norms[i](x)
            x = self._drops[i](x)
        
        # Pool to graph-level representation
        x = global_mean_pool(x, batch)
        
        return x
