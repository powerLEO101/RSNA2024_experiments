import os
import cv2
import torch
import pydicom
import numpy as np
import pandas as pd
import albumentations as A

from tqdm import tqdm
from os import environ
from torch.utils.data import Dataset
from torchvision.transforms import v2
from sklearn.model_selection import KFold
from .project_paths import base_path
from albumentations.pytorch import ToTensorV2
from .utils import display_images, scale_normalize

IS_INFER = False
df_meta_f = pd.read_csv(f'{base_path}/train_series_descriptions.csv')
df_meta_f_ = pd.read_csv(f'{base_path}/test_series_descriptions.csv')
df_label_co = pd.read_csv(f'{base_path}/train_label_coordinates.csv')
df_label_co['instance_number'] = df_label_co['instance_number'].apply(lambda x: x - 1) ### DEBUG!!!
# df_label_co['new_id'] = [f'{x}_{y}' for x, y in zip(df_label_co['study_id'], df_label_co['series_id'])]
# df_label_co = df_label_co.set_index('series_id')
kfold_random_seed = 23

def find_description(study_id, series_id):
    if IS_INFER:
        return df_meta_f_[(df_meta_f_['study_id'] == int(study_id)) & 
                    (df_meta_f_['series_id'] == int(series_id))]['series_description'].iloc[0]
    else:
        return df_meta_f[(df_meta_f['study_id'] == int(study_id)) & 
                    (df_meta_f['series_id'] == int(series_id))]['series_description'].iloc[0]

def get_df():
    df = pd.read_csv(f'{base_path}/train_clean.csv')
    df = df.fillna(-1)
    df = df.replace({'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2})
    df['filepath'] = df['study_id'].map(lambda x: f'{base_path}/train_images/{x}')
    folds = KFold(n_splits=5, shuffle=True, random_state=kfold_random_seed)
    for fold, (train_index, valid_index) in enumerate(folds.split(df)):
        df.loc[valid_index, 'fold'] = int(fold)
    return df

def get_df_infer():
    part_1 = os.listdir(f'../input/rsna-2024-lumbar-spine-degenerative-classification/test_images')
    part_1 = list(filter(lambda x: x.find('.DS') == -1, part_1))
    df = [{'study_id': x, 'filepath': f"../input/rsna-2024-lumbar-spine-degenerative-classification/test_images/{x}"} for x in part_1]
    df = pd.DataFrame(df)
    return df

def dicom_to_3d_tensors(main_folder_path):
    result = []
    desc = []
    series_id = []
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
        series_id.append(subfolder)
    return result, desc, series_id

def get_data(df, drop_rate=0.1):
    print('Loading data into RAM')
    data = {}
    for filepath, study_id in tqdm(df[['filepath', 'study_id']].values):
        if np.random.rand() < drop_rate:
            data[study_id] = filepath
        else:
            data[study_id] = dicom_to_3d_tensors(filepath)
    return data

def get_label(meta, label_name: str=None, drop_partial_label=False):
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
    label = []
    for i, name in enumerate(keys):
        if meta[name] == -1:
            if (not drop_partial_label) or label_name in name:
                label.extend([0] * 3)
            continue
        if label_name is not None and not label_name in name:
            if not drop_partial_label:
                label.extend([0] * 3)
            continue
        tmp = [0] * 3
        tmp[int(meta[name])] = 1
        label.extend(tmp)
    if drop_partial_label and len(label) < 30:
        label.extend([0] * 15)
    label = torch.tensor(label, dtype=torch.float32)
    return label

def generate_weights(meta, desc, weights=[1, 2, 4]):
    weights = {idx: x for idx, x in enumerate(weights)}
    result = []
    for one_desc in desc:
        one_desc = {'Sagittal T2/STIR': 'spinal',
                    'Sagittal T1': 'neural',
                    'Axial T2': 'subart'}.get(one_desc)
        verdict = meta[meta.keys().str.contains(one_desc)].max()
        verdict = weights[verdict]
        result.append(verdict)
    result = np.array(result)
    result = result / result.sum()
    return result
        
def get_verdict(meta, meta_file, weights=[1, 2, 4]):
    weights = {idx: x for idx, x in enumerate(weights)}
    study_id = meta['study_id']
    label_name = f"{meta['condition'].replace(' ', '_')}_{meta['level'].replace('/','_')}".lower()
    try:
        verdict = weights[meta_file[label_name]]
    except KeyError:
        verdict = 0
    return verdict


def get_df_co(df_co):
    # only sagittal view
    only_sagittal = []
    for s, id in df_co[['study_id', 'series_id']].values:
        d = find_description(s, id)
        if 'Axial' in d:
            only_sagittal.append(False)
        else:
            only_sagittal.append(True)
    df_co_only_sag = df_co[only_sagittal].reset_index(drop=True)

    new_df_co = []
    levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    for series_id, sub_df1 in df_co_only_sag.groupby('series_id'):
        for instance_number, sub_df2 in sub_df1.groupby('instance_number'):
            if len(sub_df2['condition'].unique()) != 1:
                continue
            sub_df2 = sub_df2.reset_index(drop=True)
            condition = sub_df2.iloc[0]['condition']
            coors = [[0, 0]] * 10
            have_keypoints = [False] * 10

            if 'Spinal' in condition:
                for idx, l in enumerate(levels):
                    tmp = sub_df2[sub_df2['level'] == l].reset_index(drop=True)
                    if len(tmp) == 0:
                        continue
                    tmp = tmp.iloc[0]
                    coors[idx] = [int(tmp['x']), int(tmp['y'])]
                    have_keypoints[idx] = True
            if 'Neural' in condition:
                for idx, l in enumerate(levels):
                    tmp = sub_df2[sub_df2['level'] == l].reset_index(drop=True)
                    if len(tmp) == 0:
                        continue
                    tmp = tmp.iloc[0]
                    coors[idx + 5] = [int(tmp['x']), int(tmp['y'])]
                    have_keypoints[idx + 5] = True
            
            study_id = sub_df2.iloc[0]['study_id']

            new_df_co.append([study_id, series_id, instance_number, coors, have_keypoints])

    new_df_co = pd.DataFrame(new_df_co, columns=['study_id', 'series_id', 'instance_number', 'keypoints', 'have_keypoints'])
    folds = KFold(n_splits=5, shuffle=True, random_state=kfold_random_seed)
    for fold, (train_index, valid_index) in enumerate(folds.split(new_df_co)):
        new_df_co.loc[valid_index, 'fold'] = int(fold)
    return new_df_co

class SegmentDataset(Dataset):
    def __init__(self, 
                 df,
                 df_co, 
                 data,
                 augment_level=0,
                 rough_pos_factor=0,
                 image_size=[256, 256],
                 length=25):

        super().__init__()
        self.augment_level = augment_level
        self.rough_pos_factor = rough_pos_factor
        self.df = df
        self.df_co = df_co
        self.data = data
        self.image_size = image_size
        self.length = length

        if augment_level == 0:
            self.augment = A.ReplayCompose([
                A.Resize(*self.image_size),
                ToTensorV2()],
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        elif self.augment_level == 1:
            self.augment = A.ReplayCompose([
                A.Resize(*self.image_size),
                A.Perspective(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(p=0.5, limit=(-25, 25)),
                ToTensorV2()], 
                keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.df_co)

    def __getitem__(self, idx):
        meta = dict(self.df_co.iloc[idx])

        image = self._get_slice(meta)
        result = self.augment(image=image.numpy().astype('f'), keypoints=meta['keypoints'])
        image = result['image']
        image = torch.cat([image] * 3, dim=0)
        keypoints = torch.tensor([[int(x[0]), int(x[1])] for x in result['keypoints']])
        # masks = []
        # for y, x in keypoints: # positions are inverted in numpy axis
        #     masks.append(mask_from_keypoint(x, y, self.length, *self.image_size))

        return {
            'image': image,
            'loc': keypoints,
            #'masks': masks,
            'have_keypoints': torch.tensor(meta['have_keypoints'])
        }
    
    def _get_slice(self, meta):
        volumes, desc, s_ids = self._get_data_ram_or_disk(meta)
        volume = volumes[s_ids.index(str(meta['series_id']))]
        return volume[int(meta['instance_number'])].unsqueeze(-1)
    
    def _get_data_ram_or_disk(self, meta):
        if isinstance(self.data[meta['study_id']], str):
            return dicom_to_3d_tensors(self.data[meta['study_id']])
        else:
            return self.data[meta['study_id']]