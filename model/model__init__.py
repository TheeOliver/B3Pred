"""Graph Neural Network model architectures"""

from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .gine import GINE
from .graphsage import GraphSAGE
from .graph_stack import GraphStack
from .predictor import Predictor

__all__ = [
    'GAT',
    'GCN',
    'GIN',
    'GINE',
    'GraphSAGE',
    'GraphStack',
    'Predictor',
]