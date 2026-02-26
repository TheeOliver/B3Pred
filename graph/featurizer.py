"""
Molecular Featurization Module for BBB Prediction

Converts SMILES strings to graph representations with node, edge, and graph-level features.

Node features (from PyG from_smiles + Gasteiger charges + chirality):
    - Atom type one-hot (118 dims)
    - Degree, formal charge, num Hs, radical electrons (raw)
    - Hybridization one-hot (6 dims)
    - Aromaticity, in-ring flags
    - Gasteiger partial charge (1 dim)
    - Chirality one-hot: none/CW/CCW/other (4 dims)

Edge features (enriched, one-hot encoded):
    - Bond type one-hot: single/double/triple/aromatic (4 dims)
    - Conjugation flag (1 dim)
    - Ring membership flag (1 dim)
    - Stereo one-hot: NONE/ANY/E/Z/CIS/TRANS (6 dims)
    Total: 12 dims

Graph-level descriptors appended to every node (14 dims):
    - MolLogP           — lipophilicity (key BBB passive diffusion driver)
    - MolMR             — molar refractivity
    - TPSA              — topological polar surface area (strong BBB predictor, <90 Å²)
    - MolWt             — molecular weight (Lipinski, <500 Da)
    - NumHDonors        — H-bond donors
    - NumHAcceptors     — H-bond acceptors
    - NumRotatableBonds — molecular flexibility
    - RingCount         — total ring count
    - NumAromaticRings  — aromaticity
    - FractionCSP3      — sp3 fraction (3D character)
    - BertzCT           — topological complexity
    - HeavyAtomCount    — molecular size proxy
    - Chi0v, Chi1v      — zero- and first-order connectivity indices

NODE_FEATURE_DIM  = from_smiles output + 4 (chirality) + 1 (Gasteiger) + 14 (graph) = varies by PyG version
EDGE_FEATURE_DIM  = 12
"""

import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, GraphDescriptors
from rdkit.Chem.rdchem import ChiralType, BondStereo, BondType
from typing import Optional, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────

BOND_TYPE_MAP = {
    BondType.SINGLE:   0,
    BondType.DOUBLE:   1,
    BondType.TRIPLE:   2,
    BondType.AROMATIC: 3,
}

BOND_STEREO_MAP = {
    BondStereo.STEREONONE: 0,
    BondStereo.STEREOANY:  1,
    BondStereo.STEREOE:    2,
    BondStereo.STEREOZ:    3,
    BondStereo.STEREOCIS:  4,
    BondStereo.STEREOTRANS: 5,
}

CHIRALITY_MAP = {
    ChiralType.CHI_UNSPECIFIED:    0,
    ChiralType.CHI_TETRAHEDRAL_CW:  1,
    ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    ChiralType.CHI_OTHER:          3,
}

# Number of graph-level descriptor dimensions concatenated to each node
GRAPH_DESC_DIM = 14

# Edge feature dimensionality: 4 (bond type OH) + 1 + 1 + 6 (stereo OH)
EDGE_FEATURE_DIM = 12


# ─────────────────────────────────────────────
#  Graph-level descriptor extraction
# ─────────────────────────────────────────────

def compute_graph_descriptors(mol) -> torch.Tensor:
    """
    Compute 14-dim graph-level (molecular) descriptor vector.

    These descriptors capture physicochemical properties strongly linked to
    BBB permeability (logP, TPSA, MW, HBD/HBA, etc.) and are broadcast to
    every node so the GNN has direct access to global molecular context.

    Args:
        mol: RDKit Mol object (with Gasteiger charges already computed)

    Returns:
        Float tensor of shape (GRAPH_DESC_DIM,)
    """
    try:
        mol_logp       = Descriptors.MolLogP(mol)
        mol_mr         = Descriptors.MolMR(mol)
        tpsa           = Descriptors.TPSA(mol)
        mol_wt         = Descriptors.MolWt(mol)
        num_hd         = rdMolDescriptors.CalcNumHBD(mol)
        num_ha         = rdMolDescriptors.CalcNumHBA(mol)
        num_rot        = rdMolDescriptors.CalcNumRotatableBonds(mol)
        ring_count     = mol.GetRingInfo().NumRings()
        num_arom_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        frac_csp3      = rdMolDescriptors.CalcFractionCSP3(mol)
        bertz_ct       = GraphDescriptors.BertzCT(mol)
        heavy_atom_cnt = mol.GetNumHeavyAtoms()
        chi0v          = GraphDescriptors.Chi0v(mol)
        chi1v          = GraphDescriptors.Chi1v(mol)
    except Exception:
        return torch.zeros(GRAPH_DESC_DIM, dtype=torch.float)

    return torch.tensor([
        mol_logp,
        mol_mr,
        tpsa,
        mol_wt,
        float(num_hd),
        float(num_ha),
        float(num_rot),
        float(ring_count),
        float(num_arom_rings),
        frac_csp3,
        bertz_ct,
        float(heavy_atom_cnt),
        chi0v,
        chi1v,
    ], dtype=torch.float)


# ─────────────────────────────────────────────
#  Node feature enrichment
# ─────────────────────────────────────────────

def enrich_node_features(graph, mol, graph_desc: torch.Tensor) -> None:
    """
    Append per-atom chirality, Gasteiger partial charges, and broadcast
    graph-level descriptors to every node feature vector.

    Modifies graph.x in-place.

    Args:
        graph:       PyG Data object (graph.x already populated by from_smiles)
        mol:         RDKit Mol object (Gasteiger charges already assigned)
        graph_desc:  Graph-level descriptor tensor (GRAPH_DESC_DIM,)
    """
    num_atoms = mol.GetNumAtoms()

    # — Chirality one-hot (4 dims per atom) —
    chirality_feats = torch.zeros((num_atoms, 4), dtype=torch.float)
    for idx in range(num_atoms):
        atom = mol.GetAtomWithIdx(idx)
        chiral_idx = CHIRALITY_MAP.get(atom.GetChiralTag(), 3)
        chirality_feats[idx, chiral_idx] = 1.0

    # — Gasteiger partial charge (1 dim per atom) —
    gasteiger_feats = torch.tensor(
        [mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge") for i in range(num_atoms)],
        dtype=torch.float
    ).unsqueeze(1)
    # Replace NaN/Inf that occasionally appear for exotic atoms
    gasteiger_feats = torch.nan_to_num(gasteiger_feats, nan=0.0, posinf=0.0, neginf=0.0)

    # — Graph-level descriptors broadcast to all nodes (GRAPH_DESC_DIM dims per atom) —
    graph_desc_broadcast = graph_desc.unsqueeze(0).expand(num_atoms, -1)

    # Concatenate with existing node features
    graph.x = torch.cat([graph.x, chirality_feats, gasteiger_feats, graph_desc_broadcast], dim=1)


# ─────────────────────────────────────────────
#  Edge feature enrichment
# ─────────────────────────────────────────────

def enrich_edge_features(graph, smiles: str):
    """
    Replace edge_attr with a 12-dim one-hot encoded bond feature vector.

    Features:
        - Bond type one-hot     (4 dims): single / double / triple / aromatic
        - Conjugated            (1 dim)
        - In ring               (1 dim)
        - Bond stereo one-hot   (6 dims): NONE / ANY / E / Z / CIS / TRANS

    Args:
        graph:  PyG Data object
        smiles: SMILES string

    Returns:
        Enriched graph, or None if molecule is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    edge_attr = []

    for src, dst in graph.edge_index.t().tolist():
        bond = mol.GetBondBetweenAtoms(int(src), int(dst))

        if bond is None:
            # Self-loops or missing bonds — zero vector
            edge_attr.append(torch.zeros(EDGE_FEATURE_DIM, dtype=torch.float))
            continue

        # Bond type one-hot (4 dims)
        bond_type_oh = torch.zeros(4, dtype=torch.float)
        bond_type_oh[BOND_TYPE_MAP.get(bond.GetBondType(), 0)] = 1.0

        # Conjugation + ring (2 dims)
        bond_flags = torch.tensor([
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
        ], dtype=torch.float)

        # Stereo one-hot (6 dims)
        stereo_oh = torch.zeros(6, dtype=torch.float)
        stereo_oh[BOND_STEREO_MAP.get(bond.GetStereo(), 0)] = 1.0

        edge_attr.append(torch.cat([bond_type_oh, bond_flags, stereo_oh]))

    graph.edge_attr = torch.stack(edge_attr)
    return graph, mol   # return mol too so we only parse SMILES once


# ─────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────

class MoleculeDataset(InMemoryDataset):
    """
    PyTorch Geometric Dataset for molecular graphs.

    Converts SMILES strings to graph representations with:
      - Enriched atom features (PyG defaults + chirality + Gasteiger charge)
      - Graph-level physicochemical descriptors appended to every node
      - One-hot encoded bond features
    """

    def __init__(
            self,
            data: pd.DataFrame,
            transform: Optional[callable] = None,
            pre_transform: Optional[callable] = None
    ):
        """
        Args:
            data:          DataFrame with 'SMILES' and 'target' columns
            transform:     Optional transform applied at access time
            pre_transform: Optional pre-transform applied at construction
        """
        self.df = data.reset_index(drop=True)
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(self._process())
        logger.info(f"Dataset created: {len(self)} molecules")

    def _process(self) -> List:
        data_list = []
        skipped_count = 0

        for i, row in self.df.iterrows():
            smiles  = row["SMILES"]
            target  = row["target"]

            # ── Base graph from SMILES ──────────────────────────────────────
            try:
                graph = from_smiles(smiles)
            except Exception as e:
                logger.warning(f"Skipping invalid SMILES at index {i}: {smiles}, Error: {e}")
                skipped_count += 1
                continue

            # ── Validate ────────────────────────────────────────────────────
            if graph.x is None or graph.x.size(0) == 0:
                logger.warning(f"Empty graph at index {i}: {smiles}")
                skipped_count += 1
                continue

            # ── Edge features (returns mol object to reuse) ─────────────────
            result = enrich_edge_features(graph, smiles)
            if result is None:
                logger.warning(f"Edge feature error at index {i}: {smiles}")
                skipped_count += 1
                continue
            graph, mol = result

            # ── Gasteiger charges (required before node enrichment) ─────────
            try:
                from rdkit.Chem import AllChem
                AllChem.ComputeGasteigerCharges(mol)
            except Exception as e:
                logger.warning(f"Gasteiger charge error at index {i}: {smiles}, Error: {e}")
                skipped_count += 1
                continue

            # ── Graph-level descriptors ─────────────────────────────────────
            graph_desc = compute_graph_descriptors(mol)

            # ── Node feature enrichment ─────────────────────────────────────
            graph.x = graph.x.float()
            enrich_node_features(graph, mol, graph_desc)

            # ── Edge dtype ─────────────────────────────────────────────────
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                graph.edge_attr = graph.edge_attr.float()

            # ── Store graph descriptor as separate graph-level attribute ────
            # Useful for models that want to consume it separately (e.g. concat
            # after pooling) rather than via node broadcasting.
            graph.graph_attr = graph_desc.unsqueeze(0)  # shape (1, GRAPH_DESC_DIM)

            # ── Target ─────────────────────────────────────────────────────
            graph.y = torch.tensor([target], dtype=torch.long)

            data_list.append(graph)

        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} molecules due to processing errors")

        return data_list


# ─────────────────────────────────────────────
#  Normalization helpers
# ─────────────────────────────────────────────

def compute_feature_stats(dataset: InMemoryDataset) -> Tuple[torch.Tensor, ...]:
    """
    Compute normalization statistics for node and edge features.

    Only the continuous/non-binary portions benefit from normalization; this
    function computes statistics over all dims so callers can choose which to
    apply (e.g. skip one-hot dims by passing masks downstream).

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

    x_all = torch.cat(node_features, dim=0)
    e_all  = torch.cat(edge_features, dim=0)

    x_mean = x_all.mean(dim=0)
    x_std  = x_all.std(dim=0)  + 1e-6
    e_mean = e_all.mean(dim=0)
    e_std  = e_all.std(dim=0)  + 1e-6

    logger.info("Computed feature statistics:")
    logger.info(f"  Node features: shape={x_mean.shape}, mean range=[{x_mean.min():.3f}, {x_mean.max():.3f}]")
    logger.info(f"  Edge features: shape={e_mean.shape}, mean range=[{e_mean.min():.3f}, {e_mean.max():.3f}]")

    return x_mean, x_std, e_mean, e_std


def normalize_dataset(dataset: InMemoryDataset, stats: Tuple[torch.Tensor, ...]) -> None:
    """
    Normalize dataset in-place using precomputed statistics.

    Note: One-hot encoded dimensions will be normalized too. If you want to
    preserve them as binary values, pass a feature mask to your model instead
    of normalizing the full vector here.

    Args:
        dataset: MoleculeDataset instance
        stats:   Tuple of (node_mean, node_std, edge_mean, edge_std)
    """
    x_mean, x_std, e_mean, e_std = stats

    for data in dataset:
        if data.x is not None:
            data.x = (data.x - x_mean) / x_std
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = (data.edge_attr - e_mean) / e_std

    logger.info(f"Normalized {len(dataset)} molecules")