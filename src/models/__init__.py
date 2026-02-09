"""Models package for BBB predictor."""
from src.models.base import GraphStack
from src.models.gat import GAT
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gin import GIN
from src.models.predictor import Predictor

# Model registry for easy lookup
MODEL_REGISTRY = {
    'GAT': GAT,
    'GCN': GCN,
    'GraphSAGE': GraphSAGE,
    'GIN': GIN,
}

__all__ = [
    'GraphStack',
    'GAT',
    'GCN',
    'GraphSAGE',
    'GIN',
    'Predictor',
    'MODEL_REGISTRY',
]
