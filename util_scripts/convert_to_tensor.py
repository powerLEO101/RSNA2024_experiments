import os
import sys
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd

from tqdm import tqdm
from os import environ
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import KFold

sys.path.append('..')
from core.project_paths import base_path
from core.utils import display_images

df_meta_f = pd.read_csv(f'{base_path}/train_series_descriptions.csv')
def find_description(study_id, series_id):
    return df_meta_f[(df_meta_f['study_id'] == int(study_id)) & 
                (df_meta_f['series_id'] == int(series_id))]['series_description'].iloc[0]

def get_df():
    df = pd.read_csv(f'{base_path}/train.csv')
    df = df.fillna(-1)
    df = df.replace({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})
    df['filepath'] = df['study_id'].map(lambda x: f'{base_path}/train_images/{x}')
    return df

def dicom_to_3d_tensors(main_folder_path):
    result = []
    desc = []
    study_id = main_folder_path.split('/')[-1]
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        dicom_files = [f for f in os.listdir(subfolder_path) if f.endswith('.dcm')]
        if not dicom_files:
            continue
        dicom_files.sort(key=lambda x: int(x[:-4]))
        first_slice = pydicom.dcmread(os.path.join(subfolder_path, dicom_files[0]))
        img_shape = first_slice.pixel_array.shape
        series_description = subfolder
        num_slices = len(dicom_files)
        volume = torch.zeros((num_slices, *img_shape), dtype=torch.float16)
        for i, file in enumerate(dicom_files):
            ds = pydicom.dcmread(os.path.join(subfolder_path, file))
            x = ds.pixel_array.astype(float)
            if x.shape == volume.shape[1:]:
                volume[i, :, :] = torch.tensor(x)
            else:
                volume = volume[:i]
                break
        torch.save(volume, os.path.join(subfolder_path, 'data.pt'))
        print(f"{subfolder} saved on {os.path.join(subfolder_path, 'data.pt')}")

df = get_df()
for filepath, study_id in tqdm(df[['filepath', 'study_id']].values):
    dicom_to_3d_tensors(filepath)