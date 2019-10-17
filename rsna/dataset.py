from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import numpy as np
import torch
import torch.utils.data

from .data_utils import read_dicom, get_inputs, CLASSES


cv2.setNumThreads(1)


def build_transform():
    return A.Compose([
        A.ShiftScaleRotate(
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=1.0,
            interpolation=cv2.INTER_NEAREST,
        ),
        A.Resize(512, 512),  # FIXME  some images are smaller
        A.HorizontalFlip(),
        A.RandomCrop(448, 448),
    ])


class Dataset(torch.utils.data.Dataset):
    WINDOWS = ['brain', 'blood', 'soft']

    def __init__(self, df: pd.DataFrame, root: Path):
        self.df = df
        self.root = root
        self.transform = build_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        data = read_dicom(self.root / f'{item.Image}.dcm')
        inputs = get_inputs(data, keys=self.WINDOWS)
        image = np.stack([inputs[key] for key in self.WINDOWS])
        image = np.rollaxis(image, 0, 3)
        image = self.transform(image=image)['image']
        image = np.rollaxis(image, 2, 0)
        target = np.array([item[cls] for cls in CLASSES], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(target)
