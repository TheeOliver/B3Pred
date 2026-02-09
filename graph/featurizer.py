"""
Molecular Featurization Module for BBB Prediction

Converts SMILES strings to graph representations with node and edge features.
Handles normalization and data preprocessing.
"""

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from rdkit import Chem
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MoleculeDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for molecular graphs.

    Converts SMILES strings to graph representations with enriched edge features.
    """

    def __init__(
            self,
            data: pd.DataFrame,
            transform: Optional[callable] = None,
            pre_transform: Optional[callable] = None
    ):
        """
        Initialize molecular dataset.

        Args:
            data: DataFrame with 'SMILES' and 'target' columns
            transform: Optional transform to apply to each graph
            pre_transform: Optional pre-transform to apply to each graph
        """
        self.df = data.reset_index(drop=True)
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(self._process())

        logger.info(f"Dataset created: {len(self)} molecules")

    def _process(self) -> List:
        """
        Process DataFrame to create graph data list.

        Returns:
            List of PyTorch Geometric Data objects
        """
        data_list = []
        skipped_count = 0

        for i, row in self.df.iterrows():
            smiles = row["SMILES"]
            target = row["target"]

            # Convert SMILES to graph
            try:
                graph = from_smiles(smiles)
            except Exception as e:
                logger.warning(f"Skipping invalid SMILES at index {i}: {smiles}, Error: {e}")
                skipped_count += 1
                continue

            # Add edge features
            graph = enrich_edge_features(graph, smiles)
            if graph is None:
                logger.warning(f"Edge feature error at index {i}: {smiles}")
                skipped_count += 1
                continue

            # Validate graph
            if graph.x is None or graph.x.size(0) == 0:
                logger.warning(f"Empty graph at index {i}: {smiles}")
                skipped_count += 1
                continue

            # Ensure correct data types
            graph.x = graph.x.float()
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr.float()

            # Attach target label
            graph.y = torch.tensor([target], dtype=torch.long)

            data_list.append(graph)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} molecules due to processing errors")

        return data_list


def enrich_edge_features(graph, smiles: str):
    """
    Enrich graph with chemical bond features.

    Features include:
    - Bond type (single, double, triple, aromatic)
    - Conjugation
    - Ring membership
    - Stereochemistry

    Args:
        graph: PyTorch Geometric Data object
        smiles: SMILES string for the molecule

    Returns:
        Graph with enriched edge_attr, or None if processing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edge_attr = []

    # Process each edge
    for src, dst in graph.edge_index.t().tolist():
        bond = mol.GetBondBetweenAtoms(int(src), int(dst))

        # Handle self-loops or missing bonds (shouldn't happen in valid molecules)
        if bond is None:
            edge_attr.append(torch.zeros(4, dtype=torch.float))
            continue

        # Extract bond features
        bond_type = bond.GetBondType()
        bond_type_id = {
            Chem.rdchem.BondType.SINGLE: 0,
            Chem.rdchem.BondType.DOUBLE: 1,
            Chem.rdchem.BondType.TRIPLE: 2,
            Chem.rdchem.BondType.AROMATIC: 3,
        }.get(bond_type, 0)

        feat = torch.tensor([
            bond_type_id,
            int(bond.GetIsConjugated()),
            int(bond.IsInRing()),
            int(bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE),
        ], dtype=torch.float)

        edge_attr.append(feat)

    graph.edge_attr = torch.stack(edge_attr)
    return graph


def compute_feature_stats(dataset: InMemoryDataset) -> Tuple[torch.Tensor, ...]:
    """
    Compute normalization statistics for node and edge features.

    Args:
        dataset: MoleculeDataset instance

    Returns:
        Tuple of (node_mean, node_std, edge_mean, edge_std)
    """
    node_features = []
    edge_features = []

    for data in dataset:
        if data.x is not None:
            node_features.append(data.x)
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            edge_features.append(data.edge_attr)

    # Concatenate all features
    x_all = torch.cat(node_features, dim=0)
    e_all = torch.cat(edge_features, dim=0)

    # Compute statistics
    x_mean = x_all.mean(dim=0)
    x_std = x_all.std(dim=0) + 1e-6  # Add epsilon to prevent division by zero

    e_mean = e_all.mean(dim=0)
    e_std = e_all.std(dim=0) + 1e-6

    logger.info(f"Computed feature statistics:")
    logger.info(f"  Node features: shape={x_mean.shape}, mean range=[{x_mean.min():.3f}, {x_mean.max():.3f}]")
    logger.info(f"  Edge features: shape={e_mean.shape}, mean range=[{e_mean.min():.3f}, {e_mean.max():.3f}]")

    return x_mean, x_std, e_mean, e_std


def normalize_dataset(dataset: InMemoryDataset, stats: Tuple[torch.Tensor, ...]) -> None:
    """
    Normalize dataset using precomputed statistics.

    Modifies dataset in-place.

    Args:
        dataset: MoleculeDataset instance
        stats: Tuple of (node_mean, node_std, edge_mean, edge_std)
    """
    x_mean, x_std, e_mean, e_std = stats

    for data in dataset:
        # Normalize node features
        if data.x is not None:
            data.x = (data.x - x_mean) / x_std

        # Normalize edge features
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = (data.edge_attr - e_mean) / e_std

    logger.info(f"Normalized {len(dataset)} molecules")