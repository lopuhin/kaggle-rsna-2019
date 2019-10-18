""" Data manipulation utils, many based on
https://www.kaggle.com/dcstang/see-like-a-radiologist-with-systematic-windowing
"""
from pathlib import Path
from typing import List, Dict

import attr
import numpy as np
import pandas as pd
import pydicom
import tqdm


ROOT = Path(__file__).parent.parent / 'data'
TRAIN_ROOT = ROOT / 'stage_1_train_images'
WINDOW_CONFIG = {
    # center, width
    'brain': (40, 80),
    'blood': (80, 200),
    'soft': (40, 480),
    'bone': (600, 2800),
}
CLASSES = [
    'any',
    'epidural',
    'intraparenchymal',
    'intraventricular',
    'subarachnoid',
    'subdural',
]


def read_dicom(path: Path, only_meta=False):
    return pydicom.read_file(str(path), stop_before_pixels=only_meta)


def get_inputs(data, keys: List[str] = None) -> Dict[str, np.ndarray]:
    default_window = _get_windowing(data)
    pixels = data.pixel_array
    keys = keys or WINDOW_CONFIG.keys()	
    return {
        key: _window_image(
            pixels,
            window=attr.evolve(
                default_window,
                center=c or default_window.center,
                width=w or default_window.width,
            ),
        ) for key, (c, w) in ((key, WINDOW_CONFIG[key]) for key in keys)
    }


def load_train_df():
    cached_path = ROOT / 'stage_1_train.pkl'
    if cached_path.exists():
        return pd.read_pickle(cached_path)
    else:
        print('Generating cached train dataframe')
        meta_by_image_id = {}
        for path in tqdm.tqdm(list(TRAIN_ROOT.glob('*.dcm'))):
            meta = read_dicom(path, only_meta=True)
            meta_by_image_id[path.stem] = {
                'patient_id': meta[('0010', '0020')].value,
            }
        df = pd.read_csv(ROOT / 'stage_1_train.csv')
        df[['ID', 'Image', 'Diagnosis']] = df['ID'].str.split('_', expand=True)
        df = df[['Image', 'Diagnosis', 'Label']]
        df = df.drop_duplicates()
        df = df.pivot(index='Image', columns='Diagnosis', values='Label')
        df = df.reset_index()
        df['Image'] = 'ID_' + df['Image']
        df['Patient'] = df['Image'].apply(
            lambda image_id: meta_by_image_id[image_id]['patient_id'])
        df.to_pickle(cached_path)
        return df


@attr.s(auto_attribs=True)
class Window:
    center: int
    width: int
    intercept: int
    slope: int

        
def _int_of_first(x) -> int:
    value = x.value
    if isinstance(value, pydicom.multival.MultiValue):
        if len(set(value)) != 1:
            print('ignoring multi-value', x)
        value = value[0]
    return int(value)


def _get_windowing(data) -> Window:
    return Window(
        center=_int_of_first(data[('0028','1050')]),
        width=_int_of_first(data[('0028','1051')]),
        intercept=_int_of_first(data[('0028','1052')]),
        slope=_int_of_first(data[('0028','1053')]),
    )


def _window_image(img: np.ndarray, window: Window, rescale=True) -> np.ndarray:
    img = (img * window.slope + window.intercept)
    img_min = window.center - window.width // 2
    img_max = window.center + window.width // 2
    img = np.clip(img, img_min, img_max).astype(np.float32)
    if rescale:
        img = (img - img_min) / (img_max - img_min)
    return img
