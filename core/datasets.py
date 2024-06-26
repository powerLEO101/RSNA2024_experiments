# %%
import os
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
from .project_paths import base_path
from .utils import display_images

def find_description(study_id, series_id):
    return df_meta_f[(df_meta_f['study_id'] == int(study_id)) & 
                (df_meta_f['series_id'] == int(series_id))]['series_description'].iloc[0]

def get_df():
    df = pd.read_csv(f'{base_path}/train.csv')
    df = df.fillna(-1)
    df = df.replace({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})
    df['filepath'] = df['study_id'].map(lambda x: f'{base_path}/train_images/{x}')
    folds = KFold(n_splits=5, shuffle=True, random_state=kfold_random_seed)
    for fold, (train_index, valid_index) in enumerate(folds.split(df)):
        df.loc[valid_index, 'fold'] = int(fold)
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
        if os.path.isfile(os.path.join(subfolder_path, 'data.pt')):
            volume = torch.load(os.path.join(subfolder_path, 'data.pt'))
        else:
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
        result.append(volume)
        desc.append(find_description(study_id, subfolder))
    return result, desc

def get_data(df, drop_rate=0.1):
    print('Loading data into RAM')
    data = {}
    for filepath, study_id in tqdm(df[['filepath', 'study_id']].values):
        if np.random.rand() < drop_rate:
            data[study_id] = filepath
        else:
            data[study_id] = dicom_to_3d_tensors(filepath)
    return data

def get_label(meta):
    keys = ['spinal_canal_stenosis_l1_l2',
       'spinal_canal_stenosis_l2_l3', 'spinal_canal_stenosis_l3_l4',
       'spinal_canal_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1',
       'left_neural_foraminal_narrowing_l1_l2',
       'left_neural_foraminal_narrowing_l2_l3',
       'left_neural_foraminal_narrowing_l3_l4',
       'left_neural_foraminal_narrowing_l4_l5',
       'left_neural_foraminal_narrowing_l5_s1',
       'right_neural_foraminal_narrowing_l1_l2',
       'right_neural_foraminal_narrowing_l2_l3',
       'right_neural_foraminal_narrowing_l3_l4',
       'right_neural_foraminal_narrowing_l4_l5',
       'right_neural_foraminal_narrowing_l5_s1',
       'left_subarticular_stenosis_l1_l2', 'left_subarticular_stenosis_l2_l3',
       'left_subarticular_stenosis_l3_l4', 'left_subarticular_stenosis_l4_l5',
       'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l1_l2',
       'right_subarticular_stenosis_l2_l3',
       'right_subarticular_stenosis_l3_l4',
       'right_subarticular_stenosis_l4_l5',
       'right_subarticular_stenosis_l5_s1']
    label = torch.zeros(25 * 3)
    for i, name in enumerate(keys):
        if meta[name] == -1: continue
        label[int(i * 3 + meta[name])] = 1
    return label

df_meta_f = pd.read_csv(f'{base_path}/train_series_descriptions.csv')
kfold_random_seed = 23
df = get_df()

class Normalizer(object):
    def __init__(self, n_type):
        self.n_type = n_type
        if n_type == 'scale':
            self.n_fun = self.scale_normalize
        else:
            self.n_fun = self.no_normalize
    
    def __call__(self, x):
        result = []
        for i in range(len(x)):
            result.append(self.n_fun(x[i]))
        return result
    
    def scale_normalize(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x
    
    def no_normalize(self, x):
        return x


class ImageAugmentor(object):
    def __init__(self, method):
        self.method = method
        if method == 'baseline':
            self.augment = v2.Compose([
                v2.Resize([256, 256], interpolation=cv2.INTER_CUBIC)
            ])
        else:
            self.augment = v2.Compose([v2.Identity()])
    
    def __call__(self, x):
        return self.augment(x)

class ImagePreprocessor(object):
    def __init__(self, 
                 method='baseline',
                 normalize='scale',
                 augment='baseline'):
        self.method = method
        self.normalize = Normalizer(normalize)
        self.augment = ImageAugmentor(augment)
        if method == 'baseline':
            self.preprocess = self.baseline_preprocess
        else:
            self.preprocess = self.no_preprocess
    
    def __call__(self, x, desc):
        return self.preprocess(x, desc)
    
    def no_preprocess(self, x, desc):
        return x
    
    def baseline_preprocess(self, x, desc):
        x = self.normalize(x)
        result = torch.zeros(6, 256, 256)
        if 'Sagittal T1' in desc:
            data = x[desc.index('Sagittal T1')]
            data_ = data[len(data) - 1 : len(data) + 2]
            data_ = self.augment(data_)
            result[:3, :, :] = data_
        if 'Sagittal T2/STIR' in desc:
            data = x[desc.index('Sagittal T2/STIR')]
            data_ = data[len(data) - 1 : len(data) + 2]
            data_ = self.augment(data_)
            result[3:6, :, :] = data_
        return result

class RSNADataset(Dataset):
    def __init__(self, 
                 df,
                 data,
                 method='baseline', 
                 normalize='scale', 
                 augment='baseline'):
        super().__init__()
        self.df = df
        self.data = data
        self.preprocess = ImagePreprocessor(method, normalize, augment)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx): # every subject has at least one saggital view
        meta = dict(self.df.iloc[idx])
        image, desc = self.get_data_ram_or_disk(meta)
        image = self.preprocess(image, desc)
        label = get_label(meta)
        return {
            'image': image,
            'label': label
        }
    
    def get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]
            