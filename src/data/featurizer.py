"""SMILES to molecular graph conversion for BBB prediction."""
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from typing import Optional


class MoleculeDataset(InMemoryDataset):
    """Dataset for converting SMILES strings to molecular graphs."""
    
    def __init__(
        self, 
        data: pd.DataFrame,
        smiles_column: str = "SMILES",
        target_column: str = "target",
        transform: Optional[object] = None,
        pre_transform: Optional[object] = None
    ):
        """
        Initialize the molecule dataset.
        
        Args:
            data: DataFrame containing SMILES and target columns
            smiles_column: Name of column containing SMILES strings
            target_column: Name of column containing target labels
            transform: Optional transform to apply to data
            pre_transform: Optional pre-transform to apply to data
        """
        self.df = data.reset_index(drop=True)
        self.smiles_column = smiles_column
        self.target_column = target_column
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(self._process())
    
    def _process(self):
        """Convert SMILES strings to graph objects."""
        data_list = []
        skipped = 0
        
        for i, row in self.df.iterrows():
            smiles = row[self.smiles_column]
            target = row[self.target_column]
            
            try:
                # Convert SMILES to graph
                graph = from_smiles(smiles)
                
                # Validate graph has nodes
                if graph is None or not hasattr(graph, 'x') or graph.x.shape[0] == 0:
                    print(f"Skipping SMILES at index {i}: {smiles} - No nodes generated")
                    skipped += 1
                    continue
                
                # Ensure node features are floats
                graph.x = graph.x.float()
                
                # Attach target as tensor
                graph.y = torch.tensor(target, dtype=torch.long)
                
                data_list.append(graph)
                
            except Exception as e:
                print(f"Skipping invalid SMILES at index {i}: {smiles}, Error: {e}")
                skipped += 1
                continue
        
        print(f"Processed {len(data_list)} molecules, skipped {skipped}")
        return data_list
