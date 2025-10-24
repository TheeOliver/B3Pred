import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

class MoleculeDataset(InMemoryDataset):
    def __init__(self, data: pd.DataFrame, transform=None, pre_transform=None):
        self.df = data.reset_index(drop=True)
        super().__init__(None, transform, pre_transform)
        self.data, self.slices = self.collate(self._process())

    def _process(self):
        data_list = []
        for i, row in self.df.iterrows():
            smiles = row["SMILES"]
            target = row["target"]

            # Convert SMILES to graph
            try:
                graph = from_smiles(smiles)
            except Exception as e:
                print(f"Skipping invalid SMILES at index {i}: {smiles}, Error: {e}")
                continue

            # Ensure node features are floats
            if hasattr(graph, "x") and graph.x is not None:
                graph.x = graph.x.float()

            # Attach target as tensor
            y = torch.tensor([target], dtype=torch.long)
            graph.y = y

            if(graph.x.shape[0] == 0): # Nekoi kako 'C[N+]1(C)CCCC(OC(=O)[C+](O)(c2ccccc2)c2ccccc2)C1' davaat 0 nodes
                print(graph)
                continue

            data_list.append(graph)
        return data_list
