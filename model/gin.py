from typing import Any, Dict
import torch
from torch.nn import Sequential, Linear, ReLU, ModuleList
from torch_geometric.nn import GINConv
from configs.graph_configs import GINConfig
from model.graph_stack import GraphStack

class GIN(GraphStack):
    """
    Class for the GIN based Graph Stack
    """

    def __init__(self, node_dim: int = 1, graph_layers: int = 3, graph_hidden_channels: int = 64,
                 graph_dropouts: float = 0.5, graph_norm: bool = True, model_name: str = 'GIN'):
        super(GIN, self).__init__(model_name=model_name,
                                  graph_layers=graph_layers,
                                  graph_hidden_channels=graph_hidden_channels,
                                  graph_dropouts=graph_dropouts,
                                  graph_norm=graph_norm)

        convs = []
        for layer in range(graph_layers):
            in_ch = graph_hidden_channels if layer != 0 else node_dim
            nn = Sequential(
                Linear(in_ch, graph_hidden_channels),
                ReLU(),
                Linear(graph_hidden_channels, graph_hidden_channels),
            )
            convs.append(GINConv(nn))

        self._convs = ModuleList(convs)

    @classmethod
    def from_config(cls, config: Dict[str, Any], graph_info: Dict[str, Any]) -> torch.nn.Module:
        params = {k: config[k] for k in GINConfig.hyperparameters.keys()}
        params['node_dim'] = graph_info['node_dim']
        return cls(**params)
