"""Predictor combining graph encoder and classification head."""
import torch
from torch.nn import Dropout, Linear, ModuleList, ReLU
from typing import Dict, Any
from configs.base_config import BaseConfig


class Predictor(torch.nn.Module):
    """
    Full prediction model combining a graph encoder and MLP head.
    
    The graph encoder produces graph-level embeddings, and the MLP
    head maps these to class predictions.
    """
    
    def __init__(
        self,
        graph_encoder: torch.nn.Module,
        pred_layers: int = 3,
        pred_hidden_channels: int = 64,
        num_classes: int = BaseConfig.NUM_CLASSES,
        pred_dropout: float = 0.3
    ):
        """
        Initialize the predictor.
        
        Args:
            graph_encoder: Graph neural network for encoding molecules
            pred_layers: Number of MLP layers for prediction
            pred_hidden_channels: Hidden dimension for MLP
            num_classes: Number of output classes
            pred_dropout: Dropout rate in MLP layers
        """
        super(Predictor, self).__init__()
        
        self.graph_encoder = graph_encoder
        self.pred_layers = pred_layers
        
        # Build MLP head
        layers = []
        norms = []
        drops = []
        
        for i in range(pred_layers):
            in_features = graph_encoder.output_dim if i == 0 else pred_hidden_channels
            out_features = num_classes if i == pred_layers - 1 else pred_hidden_channels
            
            layers.append(Linear(in_features, out_features))
            
            # Add activation, norm, dropout for all but last layer
            if i < pred_layers - 1:
                norms.append(torch.nn.BatchNorm1d(out_features))
                drops.append(Dropout(pred_dropout))
        
        self._layers = ModuleList(layers)
        self._norms = ModuleList(norms)
        self._drops = ModuleList(drops)
    
    def forward(self, data):
        """
        Forward pass through graph encoder and MLP head.
        
        Args:
            data: PyTorch Geometric data object
            
        Returns:
            Class logits of shape [num_graphs, num_classes]
        """
        # Get graph embeddings
        x = self.graph_encoder(data)
        
        # Apply MLP layers
        for i in range(self.pred_layers - 1):
            x = self._layers[i](x)
            x = self._norms[i](x)
            x = ReLU()(x)
            x = self._drops[i](x)
        
        # Final output layer
        x = self._layers[-1](x)
        
        return x
    
    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        node_dim: int
    ) -> torch.nn.Module:
        """
        Create predictor from configuration.
        
        Args:
            config: Configuration dictionary
            node_dim: Dimension of input node features
            
        Returns:
            Initialized Predictor model
        """
        from src.models import MODEL_REGISTRY
        
        # Get model class and create graph encoder
        model_class = MODEL_REGISTRY[config['model_name']]
        graph_encoder = model_class.from_config(config, node_dim)
        
        # Create predictor with MLP head
        predictor = cls(
            graph_encoder=graph_encoder,
            pred_layers=config['pred_layers'],
            pred_hidden_channels=config['pred_hidden_channels'],
            num_classes=BaseConfig.NUM_CLASSES,
            pred_dropout=config['pred_dropout']
        )
        
        return predictor
