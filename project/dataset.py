import os
from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import deepchem as dc


class MoleculeDataset(Dataset):
    def __init(
        self, root: str, filename: str, test=False, transform=None, pre_transform=None
    ):
        super().__init__(self, root, transform, pre_transform)
        self.test = test
        self.filename = filename

    @property
    def raw_file_names(self) -> str:
        return self.filename

    @property
    def processed_file_names(self) -> list[str]:
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f"data_test_{i}.pt" for i in list(self.data.index)]
        else:
            return [f"data_{i}.pt" for i in list(self.data.index)]
