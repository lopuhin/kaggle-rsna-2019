from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.utils.data

from .data_utils import read_dicom, get_inputs, CLASSES


class Dataset(torch.utils.data.Dataset):
    WINDOWS = ['brain', 'blood', 'soft']

    def __init__(self, df: pd.DataFrame, root: Path):
        self.df = df
        self.root = root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        data = read_dicom(self.root / f'{item.Image}.dcm')
        inputs = get_inputs(data, keys=self.WINDOWS)
        image = np.stack([inputs[key] for key in self.WINDOWS])
        target = np.array([item[cls] for cls in CLASSES], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(target)
