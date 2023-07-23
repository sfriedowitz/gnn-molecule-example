import os
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from deepchem.feat import MolGraphConvFeaturizer


class MoleculeDataset(Dataset):
    def __init__(self, root: str, filename: str, test: bool = False):
        self.root = root
        self.filename = filename
        self.test = test
        self.raw_data = pd.read_csv(self.raw_paths[0]).reset_index()
        super().__init__(root, transform=None, pre_transform=None)

    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> list[str]:
        return [self._get_processed_filename(idx) for idx in list(self.raw_data.index)]

    def download(self) -> None:
        pass

    def process(self) -> None:
        featurizer = MolGraphConvFeaturizer(use_edges=True)
        for idx, row in tqdm(self.raw_data.iterrows(), total=self.raw_data.shape[0]):
            # Featurize molecule
            mol = Chem.MolFromSmiles(row["smiles"])
            f = featurizer._featurize(mol)

            data = f.to_pyg_graph()
            data.y = self._get_label(row["HIV_active"])
            data.smiles = row["smiles"]

            processed_fname = self._get_processed_filename(idx)
            torch.save(data, os.path.join(self.processed_dir, processed_fname))

    def _get_label(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def _get_processed_filename(self, idx: int) -> str:
        return f"data_test_{idx}.pt" if self.test else f"data_{idx}.pt"

    def len(self) -> int:
        return self.raw_data.shape[0]

    def get(self, idx: int):
        processed_fname = self._get_processed_filename(idx)
        data = torch.load(os.path.join(self.processed_dir, processed_fname))
        return data
